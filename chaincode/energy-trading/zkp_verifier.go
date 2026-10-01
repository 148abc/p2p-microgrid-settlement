package main

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

// ============================================================
// Groth16 零知识证明验证器 (BN254 曲线)
// ============================================================

// ScaleFactor / FloatToField / FloatSliceToField 已移至 admm_types.go (V1 冻结)
// 验证器通过 PublicInputToFields 组装公开输入，确保三端顺序一致

// Groth16Verifier Groth16 证明验证器
type Groth16Verifier struct {
	VK          VerificationKey
	alphaG1     bn254.G1Affine
	betaG2      bn254.G2Affine
	gammaG2     bn254.G2Affine
	deltaG2     bn254.G2Affine
	icG1        []bn254.G1Affine
	initialised bool
}

// NewGroth16Verifier 从 VerificationKey 创建验证器，预解析所有曲线点
func NewGroth16Verifier(vk VerificationKey) (*Groth16Verifier, error) {
	v := &Groth16Verifier{VK: vk}

	// 反序列化验证密钥中的 G1 点 (alpha)
	if _, err := v.alphaG1.SetBytes(vk.Alpha); err != nil {
		return nil, fmt.Errorf("反序列化 alpha 失败: %w", err)
	}

	// 反序列化验证密钥中的 G2 点 (beta, gamma, delta)
	if _, err := v.betaG2.SetBytes(vk.Beta); err != nil {
		return nil, fmt.Errorf("反序列化 beta 失败: %w", err)
	}
	if _, err := v.gammaG2.SetBytes(vk.Gamma); err != nil {
		return nil, fmt.Errorf("反序列化 gamma 失败: %w", err)
	}
	if _, err := v.deltaG2.SetBytes(vk.Delta); err != nil {
		return nil, fmt.Errorf("反序列化 delta 失败: %w", err)
	}

	// 反序列化 IC 数组 (G1 点)
	v.icG1 = make([]bn254.G1Affine, len(vk.IC))
	for i, icBytes := range vk.IC {
		if _, err := v.icG1[i].SetBytes(icBytes); err != nil {
			return nil, fmt.Errorf("反序列化 IC[%d] 失败: %w", i, err)
		}
	}

	v.initialised = true
	return v, nil
}

// FloatToField / FloatSliceToField 已移至 admm_types.go (V1 冻结)

// Verify 验证 Groth16 证明
//
// Groth16 验证方程:
//
//	e(A, B) = e(α, β) · e(IC_commit, γ) · e(C, δ)
//	其中 IC_commit = IC[0] + Σ(x_i · IC[i+1])
//
// 等价的多配对检查 (Pairing Check, 乘积 = 1):
//
//	e(A, B) · e(-α, β) · e(-IC_commit, γ) · e(-C, δ) = 1
//
// 参数:
//   - proof: Groth16 证明 (A∈G1, B∈G2, C∈G1)
//   - publicInputs: 公开输入 x_1, x_2, ..., x_n (已转换为域元素)
//
// 返回:
//   - bool: 验证是否通过
//   - error: 反序列化或计算错误
func (v *Groth16Verifier) Verify(proof ZKPProof, publicInputs []*big.Int) (bool, error) {
	if !v.initialised {
		return false, fmt.Errorf("验证器未初始化")
	}

	// 检查公开输入数量是否匹配 IC 数组长度
	expectedInputs := len(v.icG1) - 1
	if len(publicInputs) != expectedInputs {
		return false, fmt.Errorf("公开输入数量不匹配: 实际 %d, 期望 %d", len(publicInputs), expectedInputs)
	}

	// ===== 1. 反序列化证明中的曲线点 =====
	var A, C bn254.G1Affine
	var B bn254.G2Affine

	if _, err := A.SetBytes(proof.Ar); err != nil {
		return false, fmt.Errorf("反序列化证明 A 失败: %w", err)
	}
	if _, err := B.SetBytes(proof.Bs); err != nil {
		return false, fmt.Errorf("反序列化证明 B 失败: %w", err)
	}
	if _, err := C.SetBytes(proof.Cr); err != nil {
		return false, fmt.Errorf("反序列化证明 C 失败: %w", err)
	}

	// ===== 2. 计算 IC commitment =====
	// IC_commit = IC[0] + x_1·IC[1] + x_2·IC[2] + ... + x_n·IC[n]
	// 用 Pippenger 批量标量乘代替逐点 ScalarMultiplication: 公开输入 588
	// 域元素时逐点相乘需要 589 次独立标量乘 (实测约 28 ms/证, 占合约侧验证
	// 成本的绝大部分), 批量 MSM 把这一步降到约 1--2 ms。
	scalars := make([]fr.Element, len(publicInputs))
	for i, x := range publicInputs {
		scalars[i].SetBigInt(x)
	}
	var icCommit bn254.G1Affine
	if len(scalars) > 0 {
		if _, err := icCommit.MultiExp(v.icG1[1:], scalars, ecc.MultiExpConfig{}); err != nil {
			return false, fmt.Errorf("IC 批量标量乘失败: %w", err)
		}
	}
	icCommit.Add(&icCommit, &v.icG1[0])

	// ===== 3. 取反（用于配对检查） =====
	// 需要: e(A,B) · e(-α,β) · e(-IC_commit,γ) · e(-C,δ) = 1
	var negAlpha, negICCommit, negC bn254.G1Affine
	negAlpha.Neg(&v.alphaG1)
	negICCommit.Neg(&icCommit)
	negC.Neg(&C)

	// ===== 4. 多配对检查 =====
	// 将 4 对 (G1, G2) 点传入，检查所有配对的乘积是否等于 1
	// 使用 gnark-crypto 的 PairingCheck 函数 (接受值类型切片)
	// 该函数内部使用优化的 multi-Miller-loop + final exponentiation
	Ps := []bn254.G1Affine{A, negAlpha, negICCommit, negC}
	Qs := []bn254.G2Affine{B, v.betaG2, v.gammaG2, v.deltaG2}

	ok, err := bn254.PairingCheck(Ps, Qs)
	if err != nil {
		return false, fmt.Errorf("配对检查失败: %w", err)
	}

	return ok, nil
}

// VerifyADMMProofV2 验证 v2 per-MG/结算级证明 (现行市场语句).
//
// 与 V1 的关系: V1 (VerifyADMMProof + PublicInput) 是 fleet 级旧语句,
// 保留供既有测试与 v1 电路部署回放; V2 是与
// zkp_circuit/circuit_v2.go 对齐的现行语句 (含 FiT 上网/柴油/η=0.95
// SOC/互斥/公开容量 RC/弃电上界/成本一致性, 字段序见
// admm_types_v2.go 的 PublicInputV2 冻结契约).
//
// 参数:
//   - proof: Groth16 证明 (A∈G1, B∈G2, C∈G1)
//   - publicInputsV2: V2 公开输入 (结构校验后按冻结顺序组装)
//   - vk: v2 电路的验证密钥 (与 V1 vk 不可混用 — 公开输入数不同:
//     V2 为 24T+12 (T=24: 588), V1 为 6+2N²T+2NT+N(N-1))
func VerifyADMMProofV2(proof ZKPProof, publicInputsV2 PublicInputV2, vk VerificationKey) (bool, error) {
	// 结构校验 + 域元素组装一次完成: 版本号/长度/非负, 且每个域元素只解析一次
	publics, err := ValidateAndParsePublicInputV2(publicInputsV2)
	if err != nil {
		return false, fmt.Errorf("V2 公开输入校验失败: %w", err)
	}
	verifier, err := NewGroth16Verifier(vk)
	if err != nil {
		return false, fmt.Errorf("创建验证器失败: %w", err)
	}
	return verifier.Verify(proof, publics)
}

// VerifyADMMProofV2With 复用已构建的验证器验证一份 V2 证明。
// 结算交易一次携带 N 份证明, 而验证器的构建要解析 VK 的全部曲线点
// (IC = 公开输入数+1 个点), 因此交易内应只构建一次并复用 —— 单笔
// N=10 结算里这一项从约 29 ms 降到约 3 ms (见 probe_vk_v2_cost_test.go)。
func VerifyADMMProofV2With(verifier *Groth16Verifier, proof ZKPProof,
	publicInputsV2 PublicInputV2) (bool, error) {
	publics, err := ValidateAndParsePublicInputV2(publicInputsV2)
	if err != nil {
		return false, fmt.Errorf("V2 公开输入校验失败: %w", err)
	}
	return verifier.Verify(proof, publics)
}

// VerifyADMMProof 验证 ADMM 终态结果的零知识证明（V1 公开输入，不耦合 ADMM 会话结构）
//
// 这是面向业务的 V1 高层接口，封装了公开输入的组装和域元素转换。
// WP2.5 起使用 PublicInput V1 冻结字段顺序，通过 PublicInputToFields 组装，
// 确保 Go 链码 / Python witness / gnark 电路 三端顺序一致。
//
// 证明内容（按部署电路如实声明）:
//
//	基础电路（PB, 如 N=3,T=4 时 24 约束）证明:
//	1. 功率平衡约束: load + curtail + loss_alloc = gen + P_dis - P_ch + P_grid + net_trade
//	2. 交易对称性: P_global[i][j][t] = -P_global[j][i][t]
//	扩展电路（PB+SOC / PB+COST）额外证明:
//	3. SOC 动态约束: SOC[t+1] = SOC[t] + P_ch·η - P_dis/η（η=1）
//	4. 充放电互斥: P_ch·z ≤ P_max, P_dis·(1-z) ≤ P_max
//	5. 成本一致性: cost_claim 与公开输入一致
//	电路不证明（诚实边界）: 最优性、ADMM 收敛、线路容量不等式、
//	P_grid 非负性、承诺绑定、价格真实性——这些需外部锚点或扩展电路。
//
// 参数:
//   - proof: Groth16 证明 (A∈G1, B∈G2, C∈G1)
//   - publicInputs: V1 公开输入（按冻结顺序）
//   - vk: Groth16 验证密钥（trusted setup 一次产出，链上固定）
func VerifyADMMProof(proof ZKPProof, publicInputs PublicInput, vk VerificationKey) (bool, error) {
	// 创建验证器
	verifier, err := NewGroth16Verifier(vk)
	if err != nil {
		return false, fmt.Errorf("创建验证器失败: %w", err)
	}

	// 按 V1 冻结顺序组装公开输入（与 PublicInputToFields 一致）
	publics := PublicInputToFields(publicInputs)

	// 执行 Groth16 验证
	return verifier.Verify(proof, publics)
}
