package main

import (
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"fmt"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/hyperledger/fabric-chaincode-go/shimtest"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// ============================================================
// SettleADMM 端到端测试（使用 MockStub）
//
// 覆盖:
//   1. TestSettleADMM_FullFlow     正常结算流程 + 余额/能源变化校验
//   2. TestSettleADMM_ReplayRejection  防重放（同 sessionID 拒绝）
//   3. TestSettleADMM_AntisymmetryRejection  反对称校验拒绝
//   4. TestSettleADMM_InvalidProofRejection  伪造证明拒绝
// ============================================================

// newMockClientIdentity 构造最小 ClientIdentity（满足接口）
func newMockClientIdentity() *mockClientIdentity {
	return &mockClientIdentity{ou: "admin"}
}

// newMockClientIdentityOU 指定 OU 的客户端身份 (用于权限测试)
func newMockClientIdentityOU(ou string) *mockClientIdentity {
	return &mockClientIdentity{ou: ou}
}

type mockClientIdentity struct {
	ou string
}

func (c *mockClientIdentity) GetID() (string, error)                             { return "admin", nil }
func (c *mockClientIdentity) GetMSPID() (string, error)                          { return "MicroGrid1MSP", nil }
func (c *mockClientIdentity) GetAttributeValue(string) (string, bool, error)     { return "", false, nil }
func (c *mockClientIdentity) AssertAttributeValue(string, string) error          { return nil }
func (c *mockClientIdentity) GetX509Certificate() (*x509.Certificate, error)     { return &x509.Certificate{Subject: pkix.Name{OrganizationalUnit: []string{c.ou}}}, nil }

// setupTestLedger 初始化测试账本（3 个微网）
func setupTestLedger(stub *shimtest.MockStub, ctx contractapi.TransactionContextInterface) {
	stub.MockTransactionStart("init-tx")
	contract := &EnergyTradingContract{}
	if err := contract.InitLedger(ctx); err != nil {
		panic(err)
	}
	stub.MockTransactionEnd("init-tx")
}

// generateTestProofAndVK 生成数学上合法的 Groth16 证明（与 zkp_real_proof_test.go 同逻辑）
func generateTestProofAndVK(numPublicInputs int) (ZKPProof, VerificationKey) {
	_, _, g1Gen, g2Gen := bn254.Generators()

	var P bn254.G1Affine
	P.ScalarMultiplication(&g1Gen, big.NewInt(42))
	var Q bn254.G2Affine
	Q.ScalarMultiplication(&g2Gen, big.NewInt(99))
	var g1Infinity bn254.G1Affine

	pBytes := P.Bytes()
	qBytes := Q.Bytes()
	infBytes := g1Infinity.Bytes()

	proof := ZKPProof{
		Ar: pBytes[:],
		Bs: qBytes[:],
		Cr: infBytes[:],
	}

	ic := make([][]byte, numPublicInputs+1)
	ic[0] = pBytes[:]
	for i := 1; i <= numPublicInputs; i++ {
		ic[i] = infBytes[:]
	}

	vk := VerificationKey{
		Alpha: infBytes[:],
		Beta:  qBytes[:],
		Gamma: qBytes[:],
		Delta: qBytes[:],
		IC:    ic,
	}
	return proof, vk
}

func mustJSON(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// buildAntisymmetricInput 构造反对称的 P_global 与 p2pPrice（测试数据工厂）
// trades: map[fromIdx]toIdx -> 时段0 买入量 (kWh)
func buildAntisymmetricInput(n, T int, price float64, trades [][3]int) ([][][]float64, [][][]float64) {
	pGlobal := make([][][]float64, n)
	p2pPrice := make([][][]float64, n)
	for i := 0; i < n; i++ {
		pGlobal[i] = make([][]float64, n)
		p2pPrice[i] = make([][]float64, n)
		for j := 0; j < n; j++ {
			pGlobal[i][j] = make([]float64, T)
			p2pPrice[i][j] = make([]float64, T)
			for k := 0; k < T; k++ {
				p2pPrice[i][j][k] = price
			}
		}
	}
	// trades: [from, to, amount]
	for _, tr := range trades {
		from, to, amt := tr[0], tr[1], float64(tr[2])
		pGlobal[from][to][0] = amt
		pGlobal[to][from][0] = -amt
	}
	return pGlobal, p2pPrice
}

// makeTestPublicInput 构造与 V1 冻结顺序对齐的测试公开输入
// V1 顺序: [sessionHash, costClaim, residualClaim, pGlobal(N*N*T), p2pPrice(N*N*T),
//           gridPrice(N*T), lossAlloc(N*T), lineCapacity(N*(N-1)), commitLoad, commitPV, commitWind]
// 注意: PGlobal/P2PPrice 必须填充与结算一致的真实交易数据 —— 链上结算从证明绑定的公开输入派生,
// 独立传入的 pGlobalJSON/p2pPriceJSON 仅作一致性校验。
func makeTestPublicInput(sessionID string, pGlobal, p2pPrice [][][]float64) PublicInput {
	n := len(pGlobal)
	T := len(pGlobal[0][0])
	flat := func(m [][][]float64) []float64 {
		out := make([]float64, 0, n*n*T)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				out = append(out, m[i][j]...)
			}
		}
		return out
	}
	return PublicInput{
		SessionHash:   SessionHashField(sessionID),
		CostClaim:     0,
		ResidualClaim: 0,
		PGlobal:       flat(pGlobal),
		P2PPrice:      flat(p2pPrice),
		GridPrice:     make([]float64, n*T),
		LossAlloc:     make([]float64, n*T),
		LineCapacity:  make([]float64, n*(n-1)),
	}
}

// newTestChaincode 创建带 MockStub 的链码（供测试用）
func newTestChaincode() (*shimtest.MockStub, contractapi.TransactionContextInterface) {
	cc, err := contractapi.NewChaincode(&EnergyTradingContract{})
	if err != nil {
		panic(err)
	}
	stub := shimtest.NewMockStub("energy-trading", cc)
	ctx := &contractapi.TransactionContext{}
	ctx.SetStub(stub)
	ctx.SetClientIdentity(newMockClientIdentity())
	return stub, ctx
}

// runTx 在 MockStub 的一个事务中执行函数（模拟一次链上交易）
var txCounter int

func runTx(stub *shimtest.MockStub, fn func() error) error {
	txCounter++
	txID := fmt.Sprintf("tx-%d", txCounter)
	stub.MockTransactionStart(txID)
	err := fn()
	stub.MockTransactionEnd(txID)
	return err
}

// TestSettleADMM_FullFlow 测试正常结算流程
func TestSettleADMM_FullFlow(t *testing.T) {
	stub, ctx := newTestChaincode()
	setupTestLedger(stub, ctx)

	contract := &EnergyTradingContract{}

	// V1 公开输入 field count: 11 + 2*N*N*T + 2*N*T + N*(N-1), 测试 N=3,T=1 → 41
	numPublicInputs := PublicInputFieldCount(3, 1)
	_, vk := generateTestProofAndVK(numPublicInputs)
	if err := runTx(stub, func() error {
		return contract.SetADMMVerificationKey(ctx, mustJSON(vk))
	}); err != nil {
		t.Fatalf("SetADMMVerificationKey 失败: %v", err)
	}

	// MG1(0) 从 MG2(1) 买 10kWh, MG1(0) 从 MG3(2) 买 5kWh (T=1)
	pGlobal, p2pPrice := buildAntisymmetricInput(3, 1, 0.5, [][3]int{{0, 1, 10}, {0, 2, 5}})

	proof, _ := generateTestProofAndVK(numPublicInputs)
	publicInput := makeTestPublicInput("session-001", pGlobal, p2pPrice)

	mgIDs := mustJSON([]string{"MG1", "MG2", "MG3"})
	pgJSON := mustJSON(pGlobal)
	ppJSON := mustJSON(p2pPrice)
	proofJSON := mustJSON(proof)
	piJSON := mustJSON(publicInput)

	if err := runTx(stub, func() error {
		return contract.SettleADMM(ctx, "session-001", mgIDs, pgJSON, ppJSON, proofJSON, piJSON)
	}); err != nil {
		t.Fatalf("SettleADMM 失败: %v", err)
	}

	// 校验:
	// MG1 买入 15 kWh * 0.5 = 7.5 元 = 750 分; 能源 +15000 Wh
	// MG2 卖出 10 kWh * 0.5 = 5 元 = 500 分; 能源 -10000 Wh
	// MG3 卖出 5 kWh * 0.5 = 2.5 元 = 250 分; 能源 -5000 Wh
	mg1, _ := contract.GetMicroGrid(ctx, "MG1")
	mg2, _ := contract.GetMicroGrid(ctx, "MG2")
	mg3, _ := contract.GetMicroGrid(ctx, "MG3")

	if mg1.Balance != 1000000-750 {
		t.Errorf("MG1 余额: 期望 %d, 实际 %d", 1000000-750, mg1.Balance)
	}
	if mg1.EnergyBalance != 100000+15000 {
		t.Errorf("MG1 能源: 期望 %d, 实际 %d", 100000+15000, mg1.EnergyBalance)
	}
	if mg2.Balance != 1000000+500 {
		t.Errorf("MG2 余额: 期望 %d, 实际 %d", 1000000+500, mg2.Balance)
	}
	if mg2.EnergyBalance != 50000-10000 {
		t.Errorf("MG2 能源: 期望 %d, 实际 %d", 50000-10000, mg2.EnergyBalance)
	}
	if mg3.Balance != 1000000+250 {
		t.Errorf("MG3 余额: 期望 %d, 实际 %d", 1000000+250, mg3.Balance)
	}
	if mg3.EnergyBalance != 80000-5000 {
		t.Errorf("MG3 能源: 期望 %d, 实际 %d", 80000-5000, mg3.EnergyBalance)
	}

	t.Logf("✅ 结算后: MG1=%d分/%dWh, MG2=%d分/%dWh, MG3=%d分/%dWh",
		mg1.Balance, mg1.EnergyBalance, mg2.Balance, mg2.EnergyBalance, mg3.Balance, mg3.EnergyBalance)
}

// TestSettleADMM_ReplayRejection 测试防重放
func TestSettleADMM_ReplayRejection(t *testing.T) {
	stub, ctx := newTestChaincode()
	setupTestLedger(stub, ctx)
	contract := &EnergyTradingContract{}

	// V1 公开输入 field count: 11 + 2*N*N*T + 2*N*T + N*(N-1), 测试 N=3,T=1 → 41
	numPublicInputs := PublicInputFieldCount(3, 1)
	_, vk := generateTestProofAndVK(numPublicInputs)
	if err := runTx(stub, func() error {
		return contract.SetADMMVerificationKey(ctx, mustJSON(vk))
	}); err != nil {
		t.Fatalf("SetADMMVerificationKey 失败: %v", err)
	}

	pGlobal, p2pPrice := buildAntisymmetricInput(3, 1, 0.5, [][3]int{{0, 1, 10}})
	proof, _ := generateTestProofAndVK(numPublicInputs)
	publicInput := makeTestPublicInput("session-r1", pGlobal, p2pPrice)

	mgIDs := mustJSON([]string{"MG1", "MG2", "MG3"})
	pgJSON := mustJSON(pGlobal)
	ppJSON := mustJSON(p2pPrice)
	proofJSON := mustJSON(proof)
	piJSON := mustJSON(publicInput)

	if err := runTx(stub, func() error {
		return contract.SettleADMM(ctx, "session-r1", mgIDs, pgJSON, ppJSON, proofJSON, piJSON)
	}); err != nil {
		t.Fatalf("第一次 SettleADMM 应成功: %v", err)
	}
	err := runTx(stub, func() error {
		return contract.SettleADMM(ctx, "session-r1", mgIDs, pgJSON, ppJSON, proofJSON, piJSON)
	})
	if err == nil {
		t.Fatal("第二次 SettleADMM（重放）应被拒绝，但未返回错误")
	}
	t.Logf("✅ 防重放生效: %v", err)
}

// TestSettleADMM_AntisymmetryRejection 测试反对称校验拒绝
func TestSettleADMM_AntisymmetryRejection(t *testing.T) {
	stub, ctx := newTestChaincode()
	setupTestLedger(stub, ctx)
	contract := &EnergyTradingContract{}

	// V1 公开输入 field count: 11 + 2*N*N*T + 2*N*T + N*(N-1), 测试 N=3,T=1 → 41
	numPublicInputs := PublicInputFieldCount(3, 1)
	_, vk := generateTestProofAndVK(numPublicInputs)
	if err := runTx(stub, func() error {
		return contract.SetADMMVerificationKey(ctx, mustJSON(vk))
	}); err != nil {
		t.Fatalf("SetADMMVerificationKey 失败: %v", err)
	}

	n := 3
	pGlobal := make([][][]float64, n)
	p2pPrice := make([][][]float64, n)
	for i := 0; i < n; i++ {
		pGlobal[i] = make([][]float64, n)
		p2pPrice[i] = make([][]float64, n)
		for j := 0; j < n; j++ {
			pGlobal[i][j] = make([]float64, 1)
			p2pPrice[i][j] = []float64{0.5}
		}
	}
	// 破坏反对称: P[0][1]=10 但 P[1][0]=0（应为 -10）
	pGlobal[0][1][0] = 10.0
	pGlobal[1][0][0] = 0.0

	proof, _ := generateTestProofAndVK(numPublicInputs)
	publicInput := makeTestPublicInput("session-anti", pGlobal, p2pPrice)

	mgIDs := mustJSON([]string{"MG1", "MG2", "MG3"})
	pgJSON := mustJSON(pGlobal)
	ppJSON := mustJSON(p2pPrice)
	proofJSON := mustJSON(proof)
	piJSON := mustJSON(publicInput)

	err := runTx(stub, func() error {
		return contract.SettleADMM(ctx, "session-anti", mgIDs, pgJSON, ppJSON, proofJSON, piJSON)
	})
	if err == nil {
		t.Fatal("非反对称 P_global 应被拒绝，但未返回错误")
	}
	t.Logf("✅ 反对称校验生效: %v", err)
}

// TestSettleADMM_InvalidProofRejection 测试伪造证明拒绝
func TestSettleADMM_InvalidProofRejection(t *testing.T) {
	stub, ctx := newTestChaincode()
	setupTestLedger(stub, ctx)
	contract := &EnergyTradingContract{}

	// V1 公开输入 field count: 11 + 2*N*N*T + 2*N*T + N*(N-1), 测试 N=3,T=1 → 41
	numPublicInputs := PublicInputFieldCount(3, 1)
	_, vk := generateTestProofAndVK(numPublicInputs)
	if err := runTx(stub, func() error {
		return contract.SetADMMVerificationKey(ctx, mustJSON(vk))
	}); err != nil {
		t.Fatalf("SetADMMVerificationKey 失败: %v", err)
	}

	pGlobal, p2pPrice := buildAntisymmetricInput(3, 1, 0.5, [][3]int{{0, 1, 10}})

	// 伪造 proof（篡改 A 点）
	forgedProof, _ := generateTestProofAndVK(numPublicInputs)
	_, _, g1Gen, _ := bn254.Generators()
	var fakeA bn254.G1Affine
	fakeA.ScalarMultiplication(&g1Gen, big.NewInt(999))
	fakeABytes := fakeA.Bytes()
	forgedProof.Ar = fakeABytes[:]

	publicInput := makeTestPublicInput("session-fake", pGlobal, p2pPrice)

	mgIDs := mustJSON([]string{"MG1", "MG2", "MG3"})
	pgJSON := mustJSON(pGlobal)
	ppJSON := mustJSON(p2pPrice)
	proofJSON := mustJSON(forgedProof)
	piJSON := mustJSON(publicInput)

	err := runTx(stub, func() error {
		return contract.SettleADMM(ctx, "session-fake", mgIDs, pgJSON, ppJSON, proofJSON, piJSON)
	})
	if err == nil {
		t.Fatal("伪造 proof 应被拒绝，但未返回错误")
	}
	t.Logf("✅ 伪造证明拒绝生效: %v", err)
}

// TestAdminAuth 验证密钥/账户注册的管理员鉴权
func TestAdminAuth(t *testing.T) {
	// 非管理员: SetADMMVerificationKey 与 RegisterMicroGrid 均应被拒绝
	nonCC, _ := contractapi.NewChaincode(&EnergyTradingContract{})
	nonStub := shimtest.NewMockStub("non-admin", nonCC)
	nonCtx := &contractapi.TransactionContext{}
	nonCtx.SetStub(nonStub)
	nonCtx.SetClientIdentity(newMockClientIdentityOU("client"))
	contract := &EnergyTradingContract{}
	vk := VerificationKey{Alpha: []byte{1}, Beta: []byte{2}, Gamma: []byte{3}, Delta: []byte{4}, IC: [][]byte{{5}}}

	if err := runTx(nonStub, func() error {
		return contract.SetADMMVerificationKey(nonCtx, mustJSON(vk))
	}); err == nil {
		t.Fatal("非管理员 SetADMMVerificationKey 应被拒绝")
	} else {
		t.Logf("✅ 非管理员 SetADMMVerificationKey 被拒绝: %v", err)
	}
	if err := runTx(nonStub, func() error {
		return contract.RegisterMicroGrid(nonCtx, "MG1", "微网1", 1, 1)
	}); err == nil {
		t.Fatal("非管理员 RegisterMicroGrid 应被拒绝")
	} else {
		t.Logf("✅ 非管理员 RegisterMicroGrid 被拒绝: %v", err)
	}

	// 管理员: 正常
	stub, ctx := newTestChaincode()
	if err := runTx(stub, func() error {
		return contract.SetADMMVerificationKey(ctx, mustJSON(vk))
	}); err != nil {
		t.Fatalf("管理员 SetADMMVerificationKey 失败: %v", err)
	}
	t.Logf("✅ 管理员 SetADMMVerificationKey 成功")
}
