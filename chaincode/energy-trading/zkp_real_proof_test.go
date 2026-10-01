package main

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
)

// ============================================================
// 真实 Groth16 证明验证测试
//
// 不使用 gnark 完整库（需要 Go 1.25+），而是利用 gnark-crypto
// 底层 BN254 曲线运算，手动构造数学上合法的 Groth16 证明。
//
// 构造原理:
//
//	Groth16 验证方程:
//	  e(A, B) · e(-α, β) · e(-IC_commit, γ) · e(-C, δ) = 1
//
//	令 α = O (G1 恒等元), C = O (G1 恒等元),
//	   IC[0] = P (任意 G1 点), IC[1..n] = O,
//	   β = γ = δ = Q (任意 G2 点),
//	   A = P, B = Q
//
//	则:
//	  IC_commit = IC[0] + Σ(x_i · IC[i+1]) = P + Σ(x_i · O) = P
//	  e(P, Q) · e(-O, Q) · e(-P, Q) · e(-O, Q)
//	  = e(P, Q) · 1 · e(P,Q)^{-1} · 1   (MillerLoop 跳过恒等元)
//	  = 1  ✓
//
//	公开输入可以是任意值（因为 IC[1..n] = O，标量乘法结果仍为 O）
// ============================================================

// generateRealProofAndVK 生成一个数学上合法的 Groth16 证明和验证密钥
//
// 返回:
//   - proof: ZKPProof (A=P, B=Q, C=O)
//   - vk: VerificationKey (α=O, β=γ=δ=Q, IC[0]=P, IC[1..n]=O)
//   - numPublicInputs: 公开输入数量
func generateRealProofAndVK(numPublicInputs int) (ZKPProof, VerificationKey) {
	// 获取 BN254 曲线的生成元
	_, _, g1Gen, g2Gen := bn254.Generators()

	// 选取标量 (非零, 非曲线阶倍数)
	scalarG1 := big.NewInt(42)
	scalarG2 := big.NewInt(99)

	// 构造非平凡 G1 点 P = 42 * G1_gen
	var P bn254.G1Affine
	P.ScalarMultiplication(&g1Gen, scalarG1)

	// 构造非平凡 G2 点 Q = 99 * G2_gen
	var Q bn254.G2Affine
	Q.ScalarMultiplication(&g2Gen, scalarG2)

	// 恒等元 (无穷远点): G1Affine{} 和 G2Affine{} 的零值
	var g1Infinity bn254.G1Affine // (0,0) = 无穷远点

	// 序列化
	pBytes := g1PointToBytes(P)
	qBytes := g2PointToBytes(Q)
	infG1Bytes := g1PointToBytes(g1Infinity)

	// 构造 Proof: A = P, B = Q, C = O
	proof := ZKPProof{
		Ar: pBytes,   // A ∈ G1 = P
		Bs: qBytes,   // B ∈ G2 = Q
		Cr: infG1Bytes, // C ∈ G1 = O (无穷远点)
	}

	// 构造 VK: α = O, β = γ = δ = Q, IC[0] = P, IC[1..n] = O
	ic := make([][]byte, numPublicInputs+1)
	ic[0] = pBytes // IC[0] = P
	for i := 1; i <= numPublicInputs; i++ {
		ic[i] = infG1Bytes // IC[1..n] = O
	}

	vk := VerificationKey{
		Alpha: infG1Bytes, // α = O
		Beta:  qBytes,     // β = Q
		Gamma: qBytes,     // γ = Q
		Delta: qBytes,     // δ = Q
		IC:    ic,         // IC[0]=P, IC[1..n]=O
	}

	return proof, vk
}

// g1PointToBytes 将 G1Affine 点序列化为压缩字节切片
func g1PointToBytes(p bn254.G1Affine) []byte {
	b := p.Bytes()
	return b[:]
}

// g2PointToBytes 将 G2Affine 点序列化为压缩字节切片
func g2PointToBytes(p bn254.G2Affine) []byte {
	b := p.Bytes()
	return b[:]
}

// ============================================================
// 测试用例
// ============================================================

// TestRealZKPVerify 直接测试 Groth16Verifier.Verify 的真实配对验证
func TestRealZKPVerify(t *testing.T) {
	// 构造证明和密钥 (2 个公开输入 → IC 有 3 个元素)
	numPublicInputs := 2
	proof, vk := generateRealProofAndVK(numPublicInputs)

	// 创建验证器
	verifier, err := NewGroth16Verifier(vk)
	if err != nil {
		t.Fatalf("NewGroth16Verifier 失败: %v", err)
	}

	// 构造公开输入 (可以是任意值, 因为 IC[1..n] = O)
	publicInputs := []*big.Int{
		FloatToField(3.14),
		FloatToField(2.718),
	}

	// 执行真实的 Groth16 配对验证
	ok, err := verifier.Verify(proof, publicInputs)
	if err != nil {
		t.Fatalf("Verify 失败: %v", err)
	}
	if !ok {
		t.Error("验证应通过: 数学上 e(P,Q)·e(-O,Q)·e(-P,Q)·e(-O,Q) = e(P,Q)·e(P,Q)^{-1} = 1")
	}

	t.Log("✅ TestRealZKPVerify 通过: 真实 BN254 配对验证成功")
	t.Logf("   证明 A (G1): %d bytes", len(proof.Ar))
	t.Logf("   证明 B (G2): %d bytes", len(proof.Bs))
	t.Logf("   证明 C (G1): %d bytes (无穷远点)", len(proof.Cr))
	t.Logf("   VK α (G1): %d bytes (无穷远点)", len(vk.Alpha))
	t.Logf("   VK β/γ/δ (G2): %d bytes each", len(vk.Beta))
	t.Logf("   VK IC 数组: %d 个元素 (IC[0]=P, 其余=O)", len(vk.IC))
}

// TestRealZKPVerifyInvalidProof 测试伪造证明应被拒绝
func TestRealZKPVerifyInvalidProof(t *testing.T) {
	numPublicInputs := 2
	proof, vk := generateRealProofAndVK(numPublicInputs)

	// 篡改证明: 用不同的 G1 点替换 A (使配对不再等于 1)
	_, _, g1Gen, _ := bn254.Generators()
	var fakeA bn254.G1Affine
	fakeA.ScalarMultiplication(&g1Gen, big.NewInt(999)) // 不同的标量
	proof.Ar = g1PointToBytes(fakeA)

	verifier, _ := NewGroth16Verifier(vk)
	publicInputs := []*big.Int{
		FloatToField(3.14),
		FloatToField(2.718),
	}

	ok, err := verifier.Verify(proof, publicInputs)
	if err != nil {
		t.Fatalf("Verify 不应返回错误: %v", err)
	}
	if ok {
		t.Error("篡改的证明不应通过验证")
	}

	t.Log("✅ TestRealZKPVerifyInvalidProof 通过: 伪造证明被正确拒绝")
}

// TestRealVerifyADMMProof 测试 VerifyADMMProof 的高层接口（V1 冻结顺序）
// 构造与 ADMM 终态数据匹配的真实证明
func TestRealVerifyADMMProof(t *testing.T) {
	// V1 测试参数: N=2, T=1
	// field count = 11 + 2*2*2*1 + 2*2*1 + 2*1 = 11 + 8 + 4 + 2 = 25
	n, T := 2, 1
	numPublicInputs := PublicInputFieldCount(n, T)

	p2pPrice := make([]float64, n*n*T) // [N][N][T] = 4
	for i := range p2pPrice {
		p2pPrice[i] = 0.5
	}
	gridPrice := make([]float64, n*T) // [N][T] = 2
	for i := range gridPrice {
		gridPrice[i] = 0.3
	}
	lossAlloc := make([]float64, n*T)    // [N][T] = 2
	lineCap := make([]float64, n*(n-1)) // [N*(N-1)] = 2

	t.Logf("V1 公开输入数量: %d (N=%d, T=%d)", numPublicInputs, n, T)

	// 生成真实证明和验证密钥
	proof, vk := generateRealProofAndVK(numPublicInputs)

	// 构造 V1 公开输入（按冻结顺序）
	publicInput := PublicInput{
		SessionHash:  SessionHashField("test-session-ok"),
		PGlobal:      make([]float64, n*n*T), // [N][N][T]
		P2PPrice:     p2pPrice,
		GridPrice:    gridPrice,
		LossAlloc:    lossAlloc,
		LineCapacity: lineCap,
	}

	// 执行真实 ZKP 验证 (不跳过!) — V1 签名：proof + publicInput + vk
	verified, err := VerifyADMMProof(proof, publicInput, vk)
	if err != nil {
		t.Fatalf("VerifyADMMProof 失败: %v", err)
	}
	if !verified {
		t.Error("VerifyADMMProof 应返回 true: 真实 BN254 配对验证通过")
	}

	t.Log("[OK] TestRealVerifyADMMProof 通过: V1 公开输入的真实 ZKP 验证成功")
	t.Logf("   session: test-session-ok, 公开输入: %d 个域元素", numPublicInputs)
}

// TestRealZKPInfinitySerialization 测试无穷远点的序列化/反序列化
func TestRealZKPInfinitySerialization(t *testing.T) {
	// G1 无穷远点
	var infG1 bn254.G1Affine
	if !infG1.IsInfinity() {
		t.Fatal("零值 G1Affine 应为无穷远点")
	}

	infBytes := g1PointToBytes(infG1)
	t.Logf("G1 无穷远点序列化: %d bytes, 首字节=0x%02x", len(infBytes), infBytes[0])

	// 反序列化
	var restored bn254.G1Affine
	if _, err := restored.SetBytes(infBytes); err != nil {
		t.Fatalf("反序列化无穷远点失败: %v", err)
	}
	if !restored.IsInfinity() {
		t.Error("反序列化后应为无穷远点")
	}

	// G2 无穷远点
	var infG2 bn254.G2Affine
	if !infG2.IsInfinity() {
		t.Fatal("零值 G2Affine 应为无穷远点")
	}

	infG2Bytes := g2PointToBytes(infG2)
	t.Logf("G2 无穷远点序列化: %d bytes, 首字节=0x%02x", len(infG2Bytes), infG2Bytes[0])

	var restoredG2 bn254.G2Affine
	if _, err := restoredG2.SetBytes(infG2Bytes); err != nil {
		t.Fatalf("反序列化 G2 无穷远点失败: %v", err)
	}
	if !restoredG2.IsInfinity() {
		t.Error("反序列化后应为无穷远点")
	}

	t.Log("✅ TestRealZKPInfinitySerialization 通过: 无穷远点序列化/反序列化正确")
}

// TestRealZKPNonInfinityPointSerialization 测试非无穷远点的序列化/反序列化
func TestRealZKPNonInfinityPointSerialization(t *testing.T) {
	_, _, g1Gen, g2Gen := bn254.Generators()

	// G1 点
	var p1 bn254.G1Affine
	p1.ScalarMultiplication(&g1Gen, big.NewInt(42))
	if p1.IsInfinity() {
		t.Fatal("42*G1_gen 不应为无穷远点")
	}

	p1Bytes := g1PointToBytes(p1)
	var restored1 bn254.G1Affine
	if _, err := restored1.SetBytes(p1Bytes); err != nil {
		t.Fatalf("反序列化 G1 点失败: %v", err)
	}
	if !restored1.Equal(&p1) {
		t.Error("反序列化后应与原点相等")
	}

	// G2 点
	var p2 bn254.G2Affine
	p2.ScalarMultiplication(&g2Gen, big.NewInt(99))
	if p2.IsInfinity() {
		t.Fatal("99*G2_gen 不应为无穷远点")
	}

	p2Bytes := g2PointToBytes(p2)
	var restored2 bn254.G2Affine
	if _, err := restored2.SetBytes(p2Bytes); err != nil {
		t.Fatalf("反序列化 G2 点失败: %v", err)
	}
	if !restored2.Equal(&p2) {
		t.Error("反序列化后应与原点相等")
	}

	t.Log("✅ TestRealZKPNonInfinityPointSerialization 通过: 非无穷远点序列化/反序列化正确")
	t.Logf("   G1 点: %d bytes, G2 点: %d bytes", len(p1Bytes), len(p2Bytes))
}
