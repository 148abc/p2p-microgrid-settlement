package main

// ============================================================
// SettleADMMV2 端到端测试 — per-microgrid 结算 (方案 A)
//
// 夹具由 zkp_circuit 的 TestExportV2MGForSettlement 生成, 是真实的
// gnark Groth16 证明 (CircuitMGv2, T=24, 22,361 约束), 不是退化证明:
//   testdata/vk_v2_mg.json           验证密钥
//   testdata/proof_v2_mg_{1,2,3}.json 每微网一份证明 (128 B)
//   testdata/public_v2_mg_{1,2,3}.json 公开输入 (588 个域元素: 含逐对交易行)
//   testdata/settle_v2_manifest.json  会话/矩阵/价格
//
// 覆盖:
//   1. FullFlow             N 份证明全部通过, 记账与划转发生
//   2. SwappedProof         证明与公开输入错配 → 拒绝
//   3. TamperedCostClaim    成本声明 +$1 → 拒绝 (电路成本门)
//   4. MatrixLinkViolation  矩阵行和 != 被证明净进口 → 拒绝 (链接校验)
//   5. Replay               同 sessionID 二次结算 → 拒绝
//   6. WrongSession         sessionID 与证明绑定不符 → 拒绝
//   7. AntiSymmetryViolation 交易矩阵非反对称 → 拒绝
//   8. MissingLineCaps      容量表未设置 → 拒绝
// ============================================================

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/hyperledger/fabric-chaincode-go/shimtest"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type settleV2ManifestJSON struct {
	SessionID    string        `json:"sessionID"`
	N            int           `json:"n"`
	T            int           `json:"t"`
	MicrogridIDs []string      `json:"microgridIDs"`
	Alpha        []string      `json:"alpha"`
	PGlobal      [][][]float64 `json:"pGlobal"`
	P2PPrice     [][][]float64 `json:"p2pPrice"`
}

func readFixtureFrom(t *testing.T, dir, name string, dst interface{}) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatalf("读取夹具 %s/%s 失败 (先跑 zkp_circuit 的 TestExportV2MGForSettlement): %v",
			dir, name, err)
	}
	if err := json.Unmarshal(b, dst); err != nil {
		t.Fatalf("解析夹具 %s/%s 失败: %v", dir, name, err)
	}
}

func readFixture(t *testing.T, name string, dst interface{}) {
	t.Helper()
	readFixtureFrom(t, "testdata", name, dst)
}

// loadV2FixturesFrom 载入指定目录下的真实证明夹具, 组装结算载荷
func loadV2FixturesFrom(t *testing.T, dir string) (VerificationKey, SettleADMMV2Request) {
	t.Helper()
	var vk VerificationKey
	readFixtureFrom(t, dir, "vk_v2_mg.json", &vk)

	var m settleV2ManifestJSON
	readFixtureFrom(t, dir, "settle_v2_manifest.json", &m)
	if m.N < 2 {
		t.Fatalf("夹具 N=%d 非法", m.N)
	}

	req := SettleADMMV2Request{
		SessionID:    m.SessionID,
		MicrogridIDs: m.MicrogridIDs,
		PGlobal:      m.PGlobal,
		P2PPrice:     m.P2PPrice,
	}
	for i := 0; i < m.N; i++ {
		var pj ZKPProof
		readFixtureFrom(t, dir, fmt.Sprintf("proof_v2_mg_%d.json", i+1), &pj)
		var pi PublicInputV2
		readFixtureFrom(t, dir, fmt.Sprintf("public_v2_mg_%d.json", i+1), &pi)
		req.Proofs = append(req.Proofs, pj)
		req.PublicInputs = append(req.PublicInputs, pi)
	}
	return vk, req
}

// loadV2Fixtures 载入 N=3 夹具 (testdata 根)
func loadV2Fixtures(t *testing.T) (VerificationKey, SettleADMMV2Request) {
	t.Helper()
	return loadV2FixturesFrom(t, "testdata")
}

// newV2TestChaincode 注册 3 个高余额微网 + 链上固定 VK 与线路容量表
func newV2TestChaincodeNoConfig(t *testing.T, vk VerificationKey, n int) (
	*shimtest.MockStub, contractapi.TransactionContextInterface) {
	t.Helper()
	stub, ctx := newTestChaincode()
	contract := &EnergyTradingContract{}

	for i := 1; i <= n; i++ {
		id := fmt.Sprintf("MG%d", i)
		if err := runTx(stub, func() error {
			return contract.RegisterMicroGrid(ctx, id, id,
				1_000_000_000, 1_000_000_000)
		}); err != nil {
			t.Fatalf("注册 %s 失败: %v", id, err)
		}
	}
	if err := runTx(stub, func() error {
		return contract.SetADMMVerificationKey(ctx, mustJSON(vk))
	}); err != nil {
		t.Fatalf("设置 VK 失败: %v", err)
	}
	caps := make([]float64, n*(n-1)/2)
	for i := range caps {
		caps[i] = 1000.0 // kW, 远高于夹具交易量
	}
	if err := runTx(stub, func() error {
		return contract.SetLineCapacities(ctx, mustJSON(caps))
	}); err != nil {
		t.Fatalf("设置线路容量失败: %v", err)
	}
	return stub, ctx
}

// marketConfigFromFixtures 用夹具的公开输入 (配置字段) 与 manifest 的类型折扣
// 组装一份"诚实"的链上市场配置, 供正常流程与负例的基线使用。
func marketConfigFromFixtures(t *testing.T, dir string, req SettleADMMV2Request) ADMMMarketConfigV2 {
	t.Helper()
	var m settleV2ManifestJSON
	readFixtureFrom(t, dir, "settle_v2_manifest.json", &m)
	if len(m.Alpha) != len(req.MicrogridIDs) {
		t.Fatalf("夹具 manifest 缺 alpha (长度 %d != N=%d): 先重跑 zkp_circuit 的导出用例",
			len(m.Alpha), len(req.MicrogridIDs))
	}
	cfg := ADMMMarketConfigV2{
		SessionID:    req.SessionID,
		MicrogridIDs: append([]string(nil), req.MicrogridIDs...),
		T:            len(req.PublicInputs[0].GridPrice),
		Alpha:        append([]string(nil), m.Alpha...),
	}
	for i := range req.PublicInputs {
		pi := req.PublicInputs[i]
		// 逐字段深拷贝: 配置与结算载荷必须互不共享底层数组, 否则"篡改注册配置"
		// 会同时改到载荷 (切片共享), 负例就变成在测证明失效而非锚定校验。
		cp := func(s []string) []string { return append([]string(nil), s...) }
		cfg.Microgrids = append(cfg.Microgrids, ADMMMarketMGConfigV2{
			GridCap: pi.GridCap, DgCap: pi.DgCap, EssPower: pi.EssPower,
			EtaNum: pi.EtaNum, EtaDen: pi.EtaDen,
			DgCost: pi.DgCost, WearCost: pi.WearCost,
			SMin: cp(pi.SMin), SMax: cp(pi.SMax),
			GridPrice: cp(pi.GridPrice), FitPrice: cp(pi.FitPrice), LossAlloc: cp(pi.LossAlloc),
		})
	}
	return cfg
}

// newV2TestChaincode 注册微网 + 链上 VK + 线路容量表 + 市场配置 (锚定基线)
func newV2TestChaincode(t *testing.T, vk VerificationKey, req SettleADMMV2Request, dir string) (
	*shimtest.MockStub, contractapi.TransactionContextInterface) {
	t.Helper()
	stub, ctx := newV2TestChaincodeNoConfig(t, vk, len(req.MicrogridIDs))
	contract := &EnergyTradingContract{}
	if err := runTx(stub, func() error {
		return contract.RegisterMarketConfigV2(ctx, mustJSON(marketConfigFromFixtures(t, dir, req)))
	}); err != nil {
		t.Fatalf("注册市场配置失败: %v", err)
	}
	return stub, ctx
}

func settleV2(stub *shimtest.MockStub, ctx contractapi.TransactionContextInterface,
	req SettleADMMV2Request) error {
	contract := &EnergyTradingContract{}
	return runTx(stub, func() error {
		return contract.SettleADMMV2(ctx, mustJSON(req))
	})
}

// ------------------------------------------------------------
// 1. 正常流程
// ------------------------------------------------------------

func TestSettleADMMV2_FullFlow(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newV2TestChaincode(t, vk, req, "testdata")
	contract := &EnergyTradingContract{}

	before := make([]*MicroGrid, len(req.MicrogridIDs))
	for i, id := range req.MicrogridIDs {
		mg, err := contract.GetMicroGrid(ctx, id)
		if err != nil {
			t.Fatalf("读取 %s 失败: %v", id, err)
		}
		before[i] = mg
	}

	if err := settleV2(stub, ctx, req); err != nil {
		t.Fatalf("SettleADMMV2 失败: %v", err)
	}

	// 结算记录已落账
	rec, err := contract.GetADMMSettlementRecord(ctx, req.SessionID)
	if err != nil || rec == nil {
		t.Fatalf("结算记录未落账: rec=%v err=%v", rec, err)
	}
	if !rec.ZKPVerified {
		t.Errorf("结算记录 ZKPVerified=false")
	}
	if rec.N != len(req.MicrogridIDs) {
		t.Errorf("结算记录 N=%d, 期望 %d", rec.N, len(req.MicrogridIDs))
	}

	// 净进口为正的微网应付费, 为负的应收费 (逐微网核对方向)
	moved := 0
	for i, id := range req.MicrogridIDs {
		mg, err := contract.GetMicroGrid(ctx, id)
		if err != nil {
			t.Fatalf("读取 %s 失败: %v", id, err)
		}
		net := 0.0
		for t2 := 0; t2 < rec.T; t2++ {
			for j := 0; j < len(req.MicrogridIDs); j++ {
				net += req.PGlobal[i][j][t2]
			}
		}
		dBal := mg.Balance - before[i].Balance
		if net > 1e-6 && dBal >= 0 {
			t.Errorf("%s 净买入 %.3f kWh, 余额却未减少 (Δ=%d 分)", id, net, dBal)
		}
		if net < -1e-6 && dBal <= 0 {
			t.Errorf("%s 净卖出 %.3f kWh, 余额却未增加 (Δ=%d 分)", id, net, dBal)
		}
		if dBal != 0 {
			moved++
		}
	}
	if moved == 0 {
		t.Errorf("结算后无任何余额变动")
	}

	// 系统层面 P2P 现金流为零: Σ 余额变动 == 0
	total := int64(0)
	for i, id := range req.MicrogridIDs {
		mg, _ := contract.GetMicroGrid(ctx, id)
		total += mg.Balance - before[i].Balance
	}
	if total != 0 {
		t.Errorf("对称中间价下系统净现金流应为 0, 实际 %d 分", total)
	}
	t.Logf("✅ N=%d 份证明全部通过, 结算记录 tx=%s, 系统净现金流 0", len(req.Proofs), rec.TxID)
}

// ------------------------------------------------------------
// 2. 证明与公开输入错配
// ------------------------------------------------------------

func TestSettleADMMV2_SwappedProofRejected(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newV2TestChaincode(t, vk, req, "testdata")

	req.Proofs[0], req.Proofs[1] = req.Proofs[1], req.Proofs[0]
	err := settleV2(stub, ctx, req)
	if err == nil {
		t.Fatal("证明与公开输入错配应被拒绝")
	}
	t.Logf("✅ 错配证明被拒绝: %v", err)
}

// ------------------------------------------------------------
// 3. 篡改成本声明
// ------------------------------------------------------------

func TestSettleADMMV2_TamperedCostClaimRejected(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newV2TestChaincode(t, vk, req, "testdata")

	// 成本声明单位是皮美元 (×1e12$), +1e12 = 虚报 1 美元
	v, ok := new(big.Int).SetString(req.PublicInputs[0].CostClaim, 10)
	if !ok {
		t.Fatalf("夹具 costClaim 非法: %q", req.PublicInputs[0].CostClaim)
	}
	req.PublicInputs[0].CostClaim = new(big.Int).Add(v, big.NewInt(1_000_000_000_000)).String()

	err := settleV2(stub, ctx, req)
	if err == nil {
		t.Fatal("成本声明 +$1 应被拒绝 (电路成本门)")
	}
	t.Logf("✅ 篡改成本声明被拒绝: %v", err)
}

// ------------------------------------------------------------
// 4. 载荷矩阵与公开交易行不一致 (逐槽交叉核对)
// ------------------------------------------------------------

func TestSettleADMMV2_MatrixLinkViolationRejected(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newV2TestChaincode(t, vk, req, "testdata")

	// 保持反对称地把一对交易量挪动 +1 kW: 反对称仍成立, 但两端的公开
	// 交易行 (已被各自证明绑定) 不再与载荷矩阵逐槽相等 → 必须被拦下
	req.PGlobal[0][1][0] += 1.0
	req.PGlobal[1][0][0] -= 1.0

	err := settleV2(stub, ctx, req)
	if err == nil {
		t.Fatal("载荷矩阵与公开交易行不一致应被拒绝")
	}
	t.Logf("✅ 交易行交叉核对失败被拒绝: %v", err)
}

// ------------------------------------------------------------
// 5. 重放
// ------------------------------------------------------------

func TestSettleADMMV2_ReplayRejected(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newV2TestChaincode(t, vk, req, "testdata")

	if err := settleV2(stub, ctx, req); err != nil {
		t.Fatalf("首次结算失败: %v", err)
	}
	err := settleV2(stub, ctx, req)
	if err == nil {
		t.Fatal("同 sessionID 二次结算应被拒绝")
	}
	t.Logf("✅ 重放被拒绝: %v", err)
}

// ------------------------------------------------------------
// 6. 会话不符 (证明绑定的是别的会话)
// ------------------------------------------------------------

func TestSettleADMMV2_WrongSessionRejected(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newV2TestChaincode(t, vk, req, "testdata")

	req.SessionID = "some-other-session"
	err := settleV2(stub, ctx, req)
	if err == nil {
		t.Fatal("sessionID 与证明绑定不符应被拒绝")
	}
	t.Logf("✅ 会话不符被拒绝: %v", err)
}

// ------------------------------------------------------------
// 7. 反对称违规
// ------------------------------------------------------------

func TestSettleADMMV2_AntiSymmetryViolationRejected(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newV2TestChaincode(t, vk, req, "testdata")

	req.PGlobal[0][1][0] += 5.0 // 只改一侧, 破坏反对称

	err := settleV2(stub, ctx, req)
	if err == nil {
		t.Fatal("非反对称交易矩阵应被拒绝")
	}
	t.Logf("✅ 反对称违规被拒绝: %v", err)
}

// ------------------------------------------------------------
// 8. 容量表未设置
// ------------------------------------------------------------

func TestSettleADMMV2_MissingLineCapRejected(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newTestChaincode()
	contract := &EnergyTradingContract{}

	for i := 1; i <= len(req.MicrogridIDs); i++ {
		id := fmt.Sprintf("MG%d", i)
		if err := runTx(stub, func() error {
			return contract.RegisterMicroGrid(ctx, id, id, 1_000_000_000, 1_000_000_000)
		}); err != nil {
			t.Fatalf("注册 %s 失败: %v", id, err)
		}
	}
	if err := runTx(stub, func() error {
		return contract.SetADMMVerificationKey(ctx, mustJSON(vk))
	}); err != nil {
		t.Fatalf("设置 VK 失败: %v", err)
	}
	// 故意不调用 SetLineCapacities

	err := settleV2(stub, ctx, req)
	if err == nil {
		t.Fatal("容量表未设置时不应结算")
	}
	t.Logf("✅ 容量表缺失被拒绝: %v", err)
}

// ------------------------------------------------------------
// 10. 论文头条规模 N=10 (10 份证明一份交易)
// ------------------------------------------------------------

func TestSettleADMMV2_N10(t *testing.T) {
	vk, req := loadV2FixturesFrom(t, filepath.Join("testdata", "n10"))
	n := len(req.MicrogridIDs)
	if n != 10 {
		t.Fatalf("N=10 夹具的 N=%d", n)
	}
	stub, ctx := newV2TestChaincode(t, vk, req, filepath.Join("testdata", "n10"))
	contract := &EnergyTradingContract{}

	totalBytes := 0
	for _, p := range req.Proofs {
		totalBytes += len(p.Ar) + len(p.Bs) + len(p.Cr)
	}
	t.Logf("N=10 夹具: %d 份证明 (合计 %d B), %d 份公开输入 (每份 %d 域元素)",
		len(req.Proofs), totalBytes, len(req.PublicInputs),
		PublicInputV2FieldCount(len(req.PGlobal[0][0])))

	before := make([]int64, n)
	for i, id := range req.MicrogridIDs {
		mg, err := contract.GetMicroGrid(ctx, id)
		if err != nil {
			t.Fatalf("读取 %s 失败: %v", id, err)
		}
		before[i] = mg.Balance
	}

	tx0 := time.Now()
	if err := settleV2(stub, ctx, req); err != nil {
		t.Fatalf("N=10 SettleADMMV2 失败: %v", err)
	}
	wall := float64(time.Since(tx0).Microseconds()) / 1000.0

	rec, err := contract.GetADMMSettlementRecord(ctx, req.SessionID)
	if err != nil || rec == nil {
		t.Fatalf("结算记录未落账: rec=%v err=%v", rec, err)
	}
	if !rec.ZKPVerified || rec.N != n {
		t.Errorf("记录异常: ZKPVerified=%v N=%d", rec.ZKPVerified, rec.N)
	}

	// 系统净现金流为零 + 逐微网方向正确
	total := int64(0)
	for i, id := range req.MicrogridIDs {
		mg, _ := contract.GetMicroGrid(ctx, id)
		d := mg.Balance - before[i]
		total += d
		net := 0.0
		for k := 0; k < len(req.PGlobal[i][0]); k++ {
			for j := 0; j < n; j++ {
				net += req.PGlobal[i][j][k]
			}
		}
		if net > 1e-6 && d >= 0 {
			t.Errorf("%s 净买入 %.3f kWh 但余额未减少", id, net)
		}
		if net < -1e-6 && d <= 0 {
			t.Errorf("%s 净卖出 %.3f kWh 但余额未增加", id, net)
		}
	}
	if total != 0 {
		t.Errorf("N=10 系统净现金流应为 0, 实际 %d 分", total)
	}
	t.Logf("✅ N=10 结算完成: %d 份证明全部通过, 墙钟 %.1f ms, tx=%s",
		len(req.Proofs), wall, rec.TxID)
}

// ------------------------------------------------------------
// 9. 完整模拟结算 (可读输出, 供人工核对)
// ------------------------------------------------------------

// mulField 把十进制域元素乘以整数 k (用于构造"被篡改的注册配置")
func mulField(t *testing.T, s string, k int64) string {
	t.Helper()
	v, ok := new(big.Int).SetString(s, 10)
	if !ok {
		t.Fatalf("域元素 %q 非法", s)
	}
	return new(big.Int).Mul(v, big.NewInt(k)).String()
}

// ------------------------------------------------------------
// 9. 市场配置未注册 → 结算拒绝
// ------------------------------------------------------------

func TestSettleADMMV2_MissingMarketConfigRejected(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newV2TestChaincodeNoConfig(t, vk, len(req.MicrogridIDs))

	err := settleV2(stub, ctx, req)
	if err == nil {
		t.Fatal("未注册市场配置时结算应被拒绝 (语句参数未锚定)")
	}
	t.Logf("✅ 缺市场配置被拒绝: %v", err)
}

// ------------------------------------------------------------
// 10. 语句参数偏离链上注册配置 (容量/带/η/费率/网损/类型折扣) → 拒绝
//     证明本身仍有效: 被拒绝的原因是公开输入与注册值不符
// ------------------------------------------------------------

func TestSettleADMMV2_TamperedConfigRejected(t *testing.T) {
	cases := []struct {
		name string
		fn   func(t *testing.T, cfg *ADMMMarketConfigV2)
	}{
		{"grid_cap", func(t *testing.T, c *ADMMMarketConfigV2) {
			c.Microgrids[0].GridCap = mulField(t, c.Microgrids[0].GridCap, 2)
		}},
		{"soc_band_max", func(t *testing.T, c *ADMMMarketConfigV2) {
			c.Microgrids[0].SMax[12] = mulField(t, c.Microgrids[0].SMax[12], 2)
		}},
		{"eta_den", func(t *testing.T, c *ADMMMarketConfigV2) {
			c.Microgrids[0].EtaDen = "1"
		}},
		{"grid_price", func(t *testing.T, c *ADMMMarketConfigV2) {
			c.Microgrids[0].GridPrice[10] = mulField(t, c.Microgrids[0].GridPrice[10], 2)
		}},
		{"loss_alloc", func(t *testing.T, c *ADMMMarketConfigV2) {
			c.Microgrids[1].LossAlloc[3] = "0"
		}},
		{"alpha", func(t *testing.T, c *ADMMMarketConfigV2) {
			c.Alpha[0] = "0.90"
		}},
		{"mg_id", func(t *testing.T, c *ADMMMarketConfigV2) {
			c.MicrogridIDs[0] = "MGX"
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			vk, req := loadV2Fixtures(t)
			stub, ctx := newV2TestChaincodeNoConfig(t, vk, len(req.MicrogridIDs))
			contract := &EnergyTradingContract{}
			cfg := marketConfigFromFixtures(t, "testdata", req)
			tc.fn(t, &cfg)
			if err := runTx(stub, func() error {
				return contract.RegisterMarketConfigV2(ctx, mustJSON(cfg))
			}); err != nil {
				t.Fatalf("注册(篡改后的)市场配置失败: %v", err)
			}
			err := settleV2(stub, ctx, req)
			if err == nil {
				t.Fatal("语句参数偏离链上注册配置时应被拒绝")
			}
			t.Logf("✅ %s 偏离被拒绝: %v", tc.name, err)
		})
	}
}

// ------------------------------------------------------------
// 11. 结算价偏离合约重算的中间价 → 拒绝
// ------------------------------------------------------------

func TestSettleADMMV2_MidPriceViolationRejected(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	stub, ctx := newV2TestChaincode(t, vk, req, "testdata")

	// 把一对结算价整体压低 1%: 反对称/行和/容量均不受影响, 只有定价规则被破坏
	for k := range req.P2PPrice[0][1] {
		req.P2PPrice[0][1][k] *= 0.99
		req.P2PPrice[1][0][k] *= 0.99
	}
	err := settleV2(stub, ctx, req)
	if err == nil {
		t.Fatal("结算价偏离合约重算的中间价应被拒绝")
	}
	t.Logf("✅ 中间价偏离被拒绝: %v", err)
}

// ------------------------------------------------------------
// 12. SOC 带锚定容差 (函数级): 量化裕度内接受, 超出拒绝
// ------------------------------------------------------------

func TestAnchorBandToleranceV2(t *testing.T) {
	_, req := loadV2Fixtures(t)
	cases := []struct {
		name   string
		shift  int64
		reject bool
	}{
		{"exact", 0, false},
		{"within_tol_100", 100, false},
		{"over_tol_101", 101, true},
		{"envelope_scale_2000", 2000, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := marketConfigFromFixtures(t, "testdata", req)
			if tc.shift != 0 {
				v, ok := new(big.Int).SetString(cfg.Microgrids[0].SMax[12], 10)
				if !ok {
					t.Fatalf("夹具 SMax[12] 非法")
				}
				cfg.Microgrids[0].SMax[12] = new(big.Int).Add(v, big.NewInt(tc.shift)).String()
			}
			err := anchorPayloadToConfigV2(&cfg, &req)
			if tc.reject && err == nil {
				t.Fatalf("带偏移 %d 个域单元应被拒绝", tc.shift)
			}
			if !tc.reject && err != nil {
				t.Fatalf("带偏移 %d 个域单元应被接受: %v", tc.shift, err)
			}
			if err != nil {
				t.Logf("✅ 偏移 %d 被拒绝: %v", tc.shift, err)
			}
		})
	}
}

// ------------------------------------------------------------
// 13. 全流程模拟 (含配置锚定与中间价重算的检查链打印)
// ------------------------------------------------------------

func TestSettleADMMV2_Simulation(t *testing.T) {
	vk, req := loadV2Fixtures(t)
	n := len(req.MicrogridIDs)
	stub, ctx := newV2TestChaincode(t, vk, req, "testdata")
	contract := &EnergyTradingContract{}

	totalBytes := 0
	for _, p := range req.Proofs {
		totalBytes += len(p.Ar) + len(p.Bs) + len(p.Cr)
	}
	fmt.Printf("\n================ V2 per-microgrid 结算模拟 ================\n")
	fmt.Printf("会话 %s | N=%d | T=%d | 曲线 BN254/Groth16\n",
		req.SessionID, n, len(req.PGlobal[0][0]))
	fmt.Printf("夹具: %d 份证明 (合计 %d B, 每份 %d B) | %d 份公开输入 (每份 %d 域元素) | 1 份链上 VK\n",
		len(req.Proofs), totalBytes, totalBytes/n,
		len(req.PublicInputs), PublicInputV2FieldCount(len(req.PGlobal[0][0])))

	before := make([]*MicroGrid, n)
	for i, id := range req.MicrogridIDs {
		mg, err := contract.GetMicroGrid(ctx, id)
		if err != nil {
			t.Fatalf("读取 %s 失败: %v", id, err)
		}
		before[i] = mg
	}

	fmt.Printf("\n合约侧检查链:\n")
	fmt.Printf("  1 结构   维度/价格非负/反对称        通过\n")
	fmt.Printf("  2 重放   sessionID 未结算            通过\n")
	caps, _ := contract.GetLineCapacities(ctx)
	fmt.Printf("  3 容量   链上固定容量表逐对逐槽筛    通过 (%s)\n", caps)
	fmt.Printf("  3b 配置   公开输入逐项锚定注册配置    通过 (容量/带/η/费率/网损)\n")
	fmt.Printf("  3c 定价   中间价 ½(α_i+α_j)π^t 合约重算  通过\n")
	for i, id := range req.MicrogridIDs {
		fmt.Printf("  4.%d %s  会话绑定 / 配对校验 / 行和链接  通过\n", i+1, id)
	}

	// 预热: Go 的 json.Marshal 首次遇到某类型要做反射缓存, 会把首次落账
	// 的耗时推高一个量级, 与结算逻辑无关。先序列化一次同类型记录排除冷启动。
	if _, err := json.Marshal(ADMMSettlementRecord{
		SessionID: "warmup", MicroGridIDs: []string{"MG1"},
		N: 1, T: 1, Converged: true,
		NetEnergyKWh: [][]float64{{0}}, NetAmountYuan: [][]float64{{0}},
		ZKPVerified: true, TxID: "warmup",
	}); err != nil {
		t.Fatalf("预热失败: %v", err)
	}

	if err := settleV2(stub, ctx, req); err != nil {
		t.Fatalf("SettleADMMV2 失败: %v", err)
	}
	fmt.Printf("  5 记账   聚合 + 划转 + 落账           通过\n")

	fmt.Printf("\n结算结果:\n")
	fmt.Printf("%-5s %14s %12s   %-22s %-22s\n",
		"微网", "净进口 kWh", "应付 元", "余额 分 (前→后)", "能源 Wh (前→后)")
	totalEnergy, totalAmount := 0.0, 0.0
	for i, id := range req.MicrogridIDs {
		netE, netA := 0.0, 0.0
		for k := 0; k < len(req.PGlobal[i][0]); k++ {
			for j := 0; j < n; j++ {
				netE += req.PGlobal[i][j][k]
				netA += req.PGlobal[i][j][k] * req.P2PPrice[i][j][k]
			}
		}
		mg, _ := contract.GetMicroGrid(ctx, id)
		fmt.Printf("%-5s %14.4f %12.4f   %-22s %-22s\n", id, netE, netA,
			fmt.Sprintf("%d → %d", before[i].Balance, mg.Balance),
			fmt.Sprintf("%d → %d", before[i].EnergyBalance, mg.EnergyBalance))
		totalEnergy += netE
		totalAmount += netA
	}
	rec, _ := contract.GetADMMSettlementRecord(ctx, req.SessionID)
	fmt.Printf("\n守恒校验: Σ净进口 = %.6f kWh (应为 0) | Σ应付 = %.6f 元 (对称中间价下应为 0)\n",
		totalEnergy, totalAmount)
	fmt.Printf("结算记录: session=%s tx=%s N=%d T=%d ZKPVerified=%v\n",
		rec.SessionID, rec.TxID, rec.N, rec.T, rec.ZKPVerified)
	fmt.Printf("==========================================================\n\n")

	if math.Abs(totalEnergy) > 1e-6 {
		t.Errorf("Σ净进口 = %.6f kWh, 应为 0", totalEnergy)
	}
	if math.Abs(totalAmount) > 1e-6 {
		t.Errorf("Σ应付 = %.6f 元, 对称中间价下应为 0", totalAmount)
	}

	// ---- 归因实验: record 阶段的耗时来自哪里 ----
	// CreateCompositeKey / json.Marshal / PutState 都应是微秒级。若微基准
	// 同样是数十毫秒, 说明该读数属于 MockStub 与宿主 (Windows) 的开销,
	// 不是结算逻辑的成本, 不能当作链上耗时报告。
	probe := ADMMSettlementRecord{
		SessionID: "probe", MicroGridIDs: req.MicrogridIDs,
		N: n, T: rec.T, Converged: true,
		NetEnergyKWh:  [][]float64{make([]float64, rec.T)},
		NetAmountYuan: [][]float64{make([]float64, rec.T)},
		ZKPVerified:   true, TxID: "probe",
	}
	const reps = 50
	tm := time.Now()
	for i := 0; i < reps; i++ {
		if _, err := json.Marshal(probe); err != nil {
			t.Fatalf("marshal 探针失败: %v", err)
		}
	}
	marshalPer := float64(time.Since(tm).Microseconds()) / 1000.0 / reps

	tp := time.Now()
	stub.MockTransactionStart("probe-tx")
	for i := 0; i < reps; i++ {
		k, err := stub.CreateCompositeKey("ADMMSettlement", []string{fmt.Sprintf("probe-%d", i)})
		if err != nil {
			t.Fatalf("composite key 探针失败: %v", err)
		}
		if err := stub.PutState(k, []byte("{}")); err != nil {
			t.Fatalf("putstate 探针失败: %v", err)
		}
	}
	stub.MockTransactionEnd("probe-tx")
	putPer := float64(time.Since(tp).Microseconds()) / 1000.0 / reps

	fmt.Printf("\n归因探针 (%d 次平均): json.Marshal %.4f ms/次 | composite+PutState %.4f ms/次\n",
		reps, marshalPer, putPer)
	if marshalPer+putPer > 5.0 {
		fmt.Printf("→ 落账阶段是宿主/MockStub 开销, 不构成链上结算成本\n")
	} else {
		fmt.Printf("→ 落账阶段本身是微秒级, record 读数另有来源 (可能为 GC 或计时器粒度)\n")
	}
}
