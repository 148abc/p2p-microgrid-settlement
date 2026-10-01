package main

import (
	"crypto/sha256"
	"encoding/json"
	"math"
	"math/big"
)

// ============================================================
// ZKP 零知识证明相关数据结构
// ============================================================

// ScaleFactor 定点数缩放因子: 10^6
// float64 值乘以此因子转为整数，作为 ZKP 电路的域元素
const ScaleFactor = 1000000.0

// BN254 标量域模数 (fr modulus)
// 注意: 该值必须与 EIP-197 / gnark-crypto 一致:
//   21888242871839275222246405745257275088548364400416034343698204186575808495617
// 此前误用 ...75804305617 (差 4190000), 导致负数 FloatToField 归约错误, 链上验证失败。
var bn254FrMod *big.Int

func init() {
	bn254FrMod, _ = new(big.Int).SetString(
		"21888242871839275222246405745257275088548364400416034343698204186575808495617",
		10,
	)
}

// ZKPProof Groth16 零知识证明
// 证明三元素：A ∈ G1, B ∈ G2, C ∈ G1
// 序列化为压缩格式的字节切片
type ZKPProof struct {
	Ar []byte `json:"ar"` // G1Affine 点 A (压缩格式, 32 bytes for BN254)
	Bs []byte `json:"bs"` // G2Affine 点 B (压缩格式, 64 bytes for BN254)
	Cr []byte `json:"cr"` // G1Affine 点 C (压缩格式, 32 bytes for BN254)
}

// VerificationKey Groth16 验证密钥
// 由可信设置(trusted setup)生成，部署时写入链码
type VerificationKey struct {
	Alpha []byte   `json:"alpha"` // G1Affine α
	Beta  []byte   `json:"beta"`  // G2Affine β
	Gamma []byte   `json:"gamma"` // G2Affine γ
	Delta []byte   `json:"delta"` // G2Affine δ
	IC    [][]byte `json:"ic"`    // []G1Affine, 长度 = 公开输入数 + 1
}

// ============================================================
// PublicInput V1 — 字段顺序自 WP2.5 起冻结
//
// 这是 Go 链码 / Python witness / gnark 电路 三方的输入契约。
// 任何字段顺序的变更必须同步三端，否则 proof 生成成功但 verifier 失败。
//
// 编码规则:
//   - 所有 float64 经 FloatToField (ScaleFactor=1e6) 转为域元素
//   - 数组按 row-major 展平
//   - session_hash = sha256(sessionID) 前 8 字节取低 43 位（见 SessionHashField），转为域元素
//   - 可选字段（commit_*）在不需要时填 0
//
// 字段顺序 (FROZEN):
//   [0] session_hash      域元素   会话绑定（链码校验 sessionID）
//   [1] cost_claim        域元素   声称系统总成本（WP5 成本一致性用，WP3 填 0）
//   [2] residual_claim    域元素   声称原始残差（WP5 收敛证明用，WP3 填 0）
//   [3] p_global          [N*N*T]  终态交易量，row-major: i*N*T + j*T + t
//   [4] p2p_price         [N*N*T]  P2P 价格，row-major
//   [5] grid_price        [N*T]    主网电价，row-major: i*T + t
//   [6] loss_alloc        [N*T]    网损分摊，row-major
//   [7] line_capacity     [N*(N-1)]线路容量，row-major
//   [8] commit_load       域元素   负荷承诺（可选，WP3 填 0）
//   [9] commit_pv         域元素   光伏承诺（可选，WP3 填 0）
//  [10] commit_wind       域元素   风电承诺（可选，WP3 填 0）
// ============================================================
type PublicInput struct {
	SessionHash    float64   `json:"sessionHash"`    // [0]  会话绑定哈希
	CostClaim      float64   `json:"costClaim"`      // [1]  声称系统成本
	ResidualClaim  float64   `json:"residualClaim"`  // [2]  声称原始残差
	PGlobal        []float64 `json:"pGlobal"`        // [3]  终态交易量 [N*N*T]
	P2PPrice       []float64 `json:"p2pPrice"`       // [4]  P2P 价格 [N*N*T]
	GridPrice      []float64 `json:"gridPrice"`      // [5]  主网电价 [N*T]
	LossAlloc      []float64 `json:"lossAlloc"`      // [6]  网损分摊 [N*T]
	LineCapacity   []float64 `json:"lineCapacity"`   // [7]  线路容量 [N*(N-1)]
	CommitLoad     float64   `json:"commitLoad"`     // [8]  负荷承诺（可选）
	CommitPV       float64   `json:"commitPV"`       // [9]  光伏承诺（可选）
	CommitWind     float64   `json:"commitWind"`     // [10] 风电承诺（可选）
}

// SessionHashField 将 sessionID 哈希为域元素（取 sha256 前 8 字节的低 43 位，作为 float64 安全往返）
// 43 位 → max ~8.8e12，经 FloatToField (×1e6) → ~8.8e18 < int64 上限 9.22e18，int64 安全
// （float64 管道下 int64 安全上限；未来 big.Int 直通可达 BN254 254 位）
func SessionHashField(sessionID string) float64 {
	h := sha256.Sum256([]byte(sessionID))
	trunc := h[:8] // 取前 8 字节
	v := new(big.Int).SetBytes(trunc)
	mask := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 43), big.NewInt(1))
	v.And(v, mask)
	return float64(v.Int64())
}

// FloatToField 将 float64 转换为有限域元素 (big.Int)
// 使用定点表示: value * 10^6 → 四舍五入 → big.Int → mod fr
func FloatToField(f float64) *big.Int {
	// 四舍五入到 1e-6 kW 分辨率 — 必须与 witness 生成端一致:
	// Python 侧用 int(round(x * 1e6)) (witness_generator_v2.py 的 _f),
	// 三端 (Python witness / gnark 电路 / Go 链码) 的口径必须相同。
	// 此前此处用 int64(scaled) 向零截断, 负数时比四舍五入小 1 个域单元,
	// 与 Python 端相差 1e-6 kW, 足以让行和链接与电路公开输入对不上。
	// 平局规则的已知差异: Python round() 半偶舍入 (half-to-even), Go
	// math.Round 半远离零 (half-away-from-zero), 仅在 x*1e6 恰落在 .5 时
	// 相差 1 个域单元。float64 物理量命中该平局的概率可忽略, 且由 witness
	// 整数预检兜底 —— 平局值会使行和/等式偏差 1e-6 而预检阈值 0.05 kW 不
	// 拦截, 但它破坏的是电路等式, 证明阶段直接失败, 会在 CI 暴露而非链上。
	scaled := math.Round(f * ScaleFactor)
	n := new(big.Int)
	if math.Abs(scaled) < 9.2e18 {
		n.SetInt64(int64(scaled))
	} else {
		// 超出 int64 的极端输入: 退回 big.Float 取整 (此时量化误差已无意义)
		bf := new(big.Float).SetFloat64(scaled)
		bf.Int(n)
	}
	if n.Sign() < 0 {
		n.Add(n, bn254FrMod)
	} else {
		n.Mod(n, bn254FrMod)
	}
	return n
}

// FloatSliceToField 将 float64 切片转换为域元素切片
func FloatSliceToField(fs []float64) []*big.Int {
	result := make([]*big.Int, len(fs))
	for i, f := range fs {
		result[i] = FloatToField(f)
	}
	return result
}

// PublicInputToFields 按 V1 冻结顺序将 PublicInput 组装为域元素切片
// 这是 verifier 和 proder 的共同入口，确保顺序一致
func PublicInputToFields(pi PublicInput) []*big.Int {
	var fields []*big.Int

	// [0] session_hash
	fields = append(fields, FloatToField(pi.SessionHash))
	// [1] cost_claim
	fields = append(fields, FloatToField(pi.CostClaim))
	// [2] residual_claim
	fields = append(fields, FloatToField(pi.ResidualClaim))
	// [3] p_global [N*N*T]
	fields = append(fields, FloatSliceToField(pi.PGlobal)...)
	// [4] p2p_price [N*N*T]
	fields = append(fields, FloatSliceToField(pi.P2PPrice)...)
	// [5] grid_price [N*T]
	fields = append(fields, FloatSliceToField(pi.GridPrice)...)
	// [6] loss_alloc [N*T]
	fields = append(fields, FloatSliceToField(pi.LossAlloc)...)
	// [7] line_capacity [N*(N-1)]
	fields = append(fields, FloatSliceToField(pi.LineCapacity)...)
	// [8] commit_load
	fields = append(fields, FloatToField(pi.CommitLoad))
	// [9] commit_pv
	fields = append(fields, FloatToField(pi.CommitPV))
	// [10] commit_wind
	fields = append(fields, FloatToField(pi.CommitWind))

	return fields
}

// PublicInputFieldCount 返回 V1 公开输入的域元素总数（用于校验）
// 固定 6 个: sessionHash, costClaim, residualClaim, commitLoad, commitPV, commitWind
// 向量: pGlobal(N*N*T) + p2pPrice(N*N*T) + gridPrice(N*T) + lossAlloc(N*T) + lineCapacity(N*(N-1))
func PublicInputFieldCount(n, t int) int {
	return 6 + 2*n*n*t + 2*n*t + n*(n-1)
}

// MarshalJSON 兼容旧 JSON 序列化（json tag 已定义）
// 为清晰起见，保留默认 JSON 序列化
func (pi PublicInput) MarshalJSON() ([]byte, error) {
	type alias PublicInput
	return json.Marshal(alias(pi))
}
