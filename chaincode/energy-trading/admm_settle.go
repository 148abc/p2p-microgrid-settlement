package main

import (
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// ============================================================
// SettleADMM — ADMM 终态链上结算（Settlement-Only）
//
// 职责（严格边界）:
//   - 接收离线 ADMM 收敛后的终态 P_global + p2p_price + ZKP 证明
//   - 验证 ZKP（调用 zkp_verifier.VerifyADMMProof，VK 链上固定）
//   - 基础校验（维度、反对称、价格非负、sessionID 防重放）
//   - 聚合 settlement（P_global → 每对 MG 净能量/净金额，链上派生，不接收外部声明）
//   - 账本结算（直接余额转账，复用 contract.go 的 MicroGrid 账户结构）
//
// 不做的事:
//   - 不做任何 ADMM 迭代、全局更新、对偶更新、收敛判定、自适应 ρ
//   - 这些只在离线 Python 跑一遍，链上零重复
//
// 单位约定:
//   P_global[i][j][t]: kW (t 时段平均功率, 每时段 1h → 能量 = P_global[i][j][t] kWh)
//   p2p_price[i][j][t]: 元/kWh
//   净能量(kWh) = Σ_t P_global[i][j][t]; 净金额(元) = Σ_t P_global[i][j][t]*p2p_price[i][j][t]
//   账本: EnergyBalance(Wh) = kWh*1000, Balance(分) = 元*100
// ============================================================

const (
	// admmEpsilon 浮点比较容差（反对称校验）
	admmEpsilon = 1e-6
	// admmVKKey 验证密钥链上存储键
	admmVKKey = "ADMM_VK"
)

// SettleTiming SettleADMM 内部各阶段耗时 (链上实测, 论文用)
type SettleTiming struct {
	JSONParse     float64 `json:"json_parse_ms"`     // 参数反序列化
	BasicValidate float64 `json:"basic_validate_ms"` // 基础校验 (维度/反对称/价格)
	ReplayCheck   float64 `json:"replay_check_ms"`   // 防重放 (账本读)
	SessionCheck  float64 `json:"session_check_ms"`  // session_hash 校验
	VKLoad        float64 `json:"vk_load_ms"`        // VK 读取 (账本读)
	ZKPVerify     float64 `json:"zkp_verify_ms"`     // Groth16 配对验证
	RebuildCheck  float64 `json:"rebuild_check_ms"`  // 证明绑定输入重建 + 一致性
	Settlement    float64 `json:"settlement_ms"`     // 聚合 + 账本结算写
	RecordSave    float64 `json:"record_save_ms"`    // 结算记录写
	Total         float64 `json:"total_ms"`          // 链码执行总耗时
}

// ADMMSettlementRecord 链上结算记录（替代 WP0 删掉的 ADMMSettlement）
type ADMMSettlementRecord struct {
	SessionID     string        `json:"sessionId"`
	MicroGridIDs  []string      `json:"microgridIds"`
	N             int           `json:"n"`
	T             int           `json:"t"`
	Converged     bool          `json:"converged"`
	NetEnergyKWh  [][]float64   `json:"netEnergyKWh"`  // [N][N] 每对 MG 净能量 (kWh, 反对称)
	NetAmountYuan [][]float64   `json:"netAmountYuan"` // [N][N] 每对 MG 净金额 (元, 反对称)
	ZKPVerified   bool      `json:"zkpVerified"`
	// ConfigAnchored 标记本次 V2 结算的语句参数已与链上注册的市场配置逐项
	// 比对并通过 (V2 路径恒为 true; V1 旧路径不写该字段)
	ConfigAnchored bool      `json:"configAnchored,omitempty"`
	SettledAt      time.Time `json:"settledAt"`
	TxID           string    `json:"txId"`
}

// timing 计时辅助
type timing struct {
	m map[string]time.Time
	t *SettleTiming
}

func newTiming() *timing { return &timing{m: map[string]time.Time{}, t: &SettleTiming{}} }

func (tm *timing) mark(name string) { tm.m[name] = time.Now() }

func (tm *timing) done(name string) {
	if s, ok := tm.m[name]; ok {
		ms := float64(time.Since(s).Microseconds()) / 1000.0
		switch name {
		case "json":
			tm.t.JSONParse = ms
		case "validate":
			tm.t.BasicValidate = ms
		case "replay":
			tm.t.ReplayCheck = ms
		case "session":
			tm.t.SessionCheck = ms
		case "vkload":
			tm.t.VKLoad = ms
		case "zkp":
			tm.t.ZKPVerify = ms
		case "rebuild":
			tm.t.RebuildCheck = ms
		case "settle":
			tm.t.Settlement = ms
		case "save":
			tm.t.RecordSave = ms
		}
	}
}

func (tm *timing) total() *SettleTiming {
	tm.t.Total = tm.t.JSONParse + tm.t.BasicValidate + tm.t.ReplayCheck +
		tm.t.SessionCheck + tm.t.VKLoad + tm.t.ZKPVerify + tm.t.RebuildCheck +
		tm.t.Settlement + tm.t.RecordSave
	return tm.t
}

// SettleADMM 接收离线 ADMM 终态，验证 ZKP，完成账上结算。
//
// 参数:
//   - sessionID: 会话唯一标识（proof 绑定，防重放）
//   - microgridIDsJSON: ["MG1","MG2","MG3"]
//   - pGlobalJSON: [N][N][T] 终态交易量 (kW, 反对称 P_global[i][j] = -P_global[j][i])
//   - p2pPriceJSON: [N][N][T] P2P 价格 (元/kWh)
//   - proofJSON: Groth16 证明
//   - publicInputJSON: ZKP 公开输入
//
// VK（验证密钥）链上固定（Ledger 存储，SetADMMVerificationKey 一次性设置），不随交易传。
func (c *EnergyTradingContract) SettleADMM(
	ctx contractapi.TransactionContextInterface,
	sessionID string,
	microgridIDsJSON string,
	pGlobalJSON string,
	p2pPriceJSON string,
	proofJSON string,
	publicInputJSON string,
) error {
	// 耗时拆分 (链上实测)
	tm := newTiming()
	tm.mark("json")

	// ===== 1. 反序列化 =====
	var microgridIDs []string
	if err := json.Unmarshal([]byte(microgridIDsJSON), &microgridIDs); err != nil {
		return fmt.Errorf("解析 microgridIDs 失败: %w", err)
	}
	var pGlobal [][][]float64
	if err := json.Unmarshal([]byte(pGlobalJSON), &pGlobal); err != nil {
		return fmt.Errorf("解析 pGlobal 失败: %w", err)
	}
	var p2pPrice [][][]float64
	if err := json.Unmarshal([]byte(p2pPriceJSON), &p2pPrice); err != nil {
		return fmt.Errorf("解析 p2pPrice 失败: %w", err)
	}
	var proof ZKPProof
	if err := json.Unmarshal([]byte(proofJSON), &proof); err != nil {
		return fmt.Errorf("解析 proof 失败: %w", err)
	}
	var publicInput PublicInput
	if err := json.Unmarshal([]byte(publicInputJSON), &publicInput); err != nil {
		return fmt.Errorf("解析 publicInput 失败: %w", err)
	}

	n := len(microgridIDs)
	tm.done("json")

	// ===== 2. 基础校验 =====
	tm.mark("validate")
	if err := validateADMMFinalState(microgridIDs, pGlobal, p2pPrice); err != nil {
		return err
	}
	tm.done("validate")

	// sessionID 防重放：同一 sessionID 不得重复结算
	tm.mark("replay")
	if existing, _ := c.getADMMSettlementRecord(ctx, sessionID); existing != nil {
		return fmt.Errorf("sessionID %s 已结算，禁止重放", sessionID)
	}
	tm.done("replay")

	// ===== 3. ZKP 验证 =====
	// 校验 proof 绑定的 session_hash 与传入 sessionID 一致（防重放/跨会话复用）
	tm.mark("session")
	expectedHash := SessionHashField(sessionID)
	if expectedHash != publicInput.SessionHash {
		return fmt.Errorf("session_hash 不匹配: proof 绑定的会话与 sessionID 不一致 (期望 %.0f, 实际 %.0f)", expectedHash, publicInput.SessionHash)
	}
	tm.done("session")

	tm.mark("vkload")
	vk, err := c.getADMMVK(ctx)
	if err != nil {
		return fmt.Errorf("加载验证密钥失败: %w", err)
	}
	tm.done("vkload")

	tm.mark("zkp")
	zkpVerified, err := VerifyADMMProof(proof, publicInput, vk)
	if err != nil {
		return fmt.Errorf("ZKP 验证出错: %w", err)
	}
	if !zkpVerified {
		return fmt.Errorf("ZKP 验证失败: 证明无效")
	}
	tm.done("zkp")

	// ===== 4. 聚合 settlement（从证明绑定的公开输入派生，不信任独立参数） =====
	// 关键安全点: 结算金额必须来自 ZKP 证明覆盖的公开输入 (publicInput.PGlobal/P2PPrice),
	// 而不是调用方单独传入的 pGlobalJSON/p2pPriceJSON —— 否则证明与结算数据可脱钩。
	tm.mark("rebuild")
	provenPGlobal, err := rebuildMatrix(publicInput.PGlobal, n)
	if err != nil {
		return fmt.Errorf("公开输入 pGlobal 维度非法: %w", err)
	}
	provenP2PPrice, err := rebuildMatrix(publicInput.P2PPrice, n)
	if err != nil {
		return fmt.Errorf("公开输入 p2pPrice 维度非法: %w", err)
	}
	// 对证明绑定的数据做基础校验（维度/反对称/价格非负）
	if err := validateADMMFinalState(microgridIDs, provenPGlobal, provenP2PPrice); err != nil {
		return fmt.Errorf("证明绑定的公开输入校验失败: %w", err)
	}
	// 防御性一致性校验: 独立传入的 pGlobal/p2pPrice 必须与证明绑定数据一致
	if !matricesEqual(pGlobal, provenPGlobal) {
		return fmt.Errorf("pGlobalJSON 与证明绑定的公开输入不一致，结算被拒绝")
	}
	if !matricesEqual(p2pPrice, provenP2PPrice) {
		return fmt.Errorf("p2pPriceJSON 与证明绑定的公开输入不一致，结算被拒绝")
	}

	tm.done("rebuild")

	netEnergy, netAmount := aggregateSettlement(provenPGlobal, provenP2PPrice, n)

	// ===== 5. 账本结算 =====
	tm.mark("settle")
	if err := c.executeSettlement(ctx, microgridIDs, netEnergy, netAmount, n); err != nil {
		return fmt.Errorf("账本结算失败: %w", err)
	}
	tm.done("settle")

	// ===== 6. 记录结算 + 事件 =====
	tm.mark("save")
	txTimestamp, _ := ctx.GetStub().GetTxTimestamp()
	txID := ctx.GetStub().GetTxID()
	record := ADMMSettlementRecord{
		SessionID:     sessionID,
		MicroGridIDs:   microgridIDs,
		N:             n,
		T:             len(provenPGlobal[0][0]),
		Converged:     true,
		NetEnergyKWh:  netEnergy,
		NetAmountYuan: netAmount,
		ZKPVerified:   zkpVerified,
		SettledAt:     time.Unix(txTimestamp.Seconds, int64(txTimestamp.Nanos)),
		TxID:          txID,
	}
	if err := c.saveADMMSettlementRecord(ctx, record); err != nil {
		return fmt.Errorf("保存结算记录失败: %w", err)
	}
	tm.done("save")

	// 耗时拆分: 不写入账本状态 (跨 peer 执行耗时不同会破坏背书一致性, 导致交易被 vscc 判无效)
	// 通过事件 + peer 日志输出 (事件不影响状态一致性)
	tmTotal := tm.total()
	fmt.Printf("SETTLE_TIMING session=%s json=%.3f validate=%.3f replay=%.3f session_check=%.3f vk_load=%.3f zkp=%.3f rebuild=%.3f settle=%.3f save=%.3f total=%.3f ms\n",
		sessionID, tmTotal.JSONParse, tmTotal.BasicValidate, tmTotal.ReplayCheck,
		tmTotal.SessionCheck, tmTotal.VKLoad, tmTotal.ZKPVerify, tmTotal.RebuildCheck,
		tmTotal.Settlement, tmTotal.RecordSave, tmTotal.Total)

	// 发出事件（供链下索引; 注意: 事件在提案响应中, 不能含跨 peer 不一致的数据）
	eventPayload, _ := json.Marshal(map[string]interface{}{
		"sessionId":   sessionID,
		"n":           n,
		"zkpVerified": zkpVerified,
		"txId":        txID,
	})
	ctx.GetStub().SetEvent("SettleADMM", eventPayload)

	return nil
}

// SetADMMVerificationKey 一次性设置 ZKP 验证密钥（管理员调用，VK 链上固定）
func (c *EnergyTradingContract) SetADMMVerificationKey(
	ctx contractapi.TransactionContextInterface,
	vkJSON string,
) error {
	// 权限控制: 仅管理员 (证书 OU=admin) 可设置/替换验证密钥
	if err := requireAdmin(ctx); err != nil {
		return err
	}
	var vk VerificationKey
	if err := json.Unmarshal([]byte(vkJSON), &vk); err != nil {
		return fmt.Errorf("解析 VK 失败: %w", err)
	}
	vkBytes, _ := json.Marshal(vk)
	return ctx.GetStub().PutState(admmVKKey, vkBytes)
}

// requireAdmin 校验调用者为任一组织的管理员（证书 Subject OU=admin）
func requireAdmin(ctx contractapi.TransactionContextInterface) error {
	cert, err := ctx.GetClientIdentity().GetX509Certificate()
	if err != nil {
		return fmt.Errorf("获取客户端证书失败: %w", err)
	}
	for _, ou := range cert.Subject.OrganizationalUnit {
		if ou == "admin" {
			return nil
		}
	}
	return fmt.Errorf("权限不足: 仅管理员 (OU=admin) 可调用该接口，实际 OU=%v", cert.Subject.OrganizationalUnit)
}

// GetADMMVerificationKey 查询当前链上固定的验证密钥
func (c *EnergyTradingContract) GetADMMVerificationKey(
	ctx contractapi.TransactionContextInterface,
) (string, error) {
	vkJSON, err := ctx.GetStub().GetState(admmVKKey)
	if err != nil {
		return "", fmt.Errorf("读取 VK 失败: %w", err)
	}
	if vkJSON == nil {
		return "", fmt.Errorf("验证密钥未设置")
	}
	return string(vkJSON), nil
}

// GetADMMSettlementRecord 查询结算记录
func (c *EnergyTradingContract) GetADMMSettlementRecord(
	ctx contractapi.TransactionContextInterface,
	sessionID string,
) (*ADMMSettlementRecord, error) {
	return c.getADMMSettlementRecord(ctx, sessionID)
}

// ============================================================
// 内部：基础校验
// ============================================================

// validateADMMFinalState 校验终态维度、反对称、价格非负、微网存在
func validateADMMFinalState(microgridIDs []string, pGlobal, p2pPrice [][][]float64) error {
	n := len(microgridIDs)
	if n < 2 {
		return fmt.Errorf("微网数量不足: 需要 ≥2, 实际 %d", n)
	}
	// 维度校验
	if len(pGlobal) != n || len(p2pPrice) != n {
		return fmt.Errorf("pGlobal/p2pPrice 第一维不匹配: 期望 %d, 实际 %d/%d", n, len(pGlobal), len(p2pPrice))
	}
	t := 0
	for i := 0; i < n; i++ {
		if len(pGlobal[i]) != n || len(p2pPrice[i]) != n {
			return fmt.Errorf("pGlobal/p2pPrice 第二维不匹配: MG[%d] 实际 %d/%d", i, len(pGlobal[i]), len(p2pPrice[i]))
		}
		for j := 0; j < n; j++ {
			if i == 0 && j == 0 {
				t = len(pGlobal[0][0])
				if t == 0 {
					return fmt.Errorf("时间槽数量 T 不能为 0")
				}
			}
			if len(pGlobal[i][j]) != t || len(p2pPrice[i][j]) != t {
				return fmt.Errorf("pGlobal/p2pPrice 第三维不匹配: MG[%d][%d] 实际 %d/%d, 期望 %d", i, j, len(pGlobal[i][j]), len(p2pPrice[i][j]), t)
			}
			// 价格非负
			for k := 0; k < t; k++ {
				if p2pPrice[i][j][k] < 0 {
					return fmt.Errorf("p2pPrice[%d][%d][%d] = %.4f 为负，价格必须非负", i, j, k, p2pPrice[i][j][k])
				}
			}
		}
	}
	// 反对称校验: P_global[i][j][t] == -P_global[j][i][t]
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			for k := 0; k < t; k++ {
				if math.Abs(pGlobal[i][j][k]+pGlobal[j][i][k]) > admmEpsilon {
					return fmt.Errorf("P_global 反对称不满足: P[%d][%d][%d]=%.6f, P[%d][%d][%d]=%.6f",
						i, j, k, pGlobal[i][j][k], j, i, k, pGlobal[j][i][k])
				}
			}
		}
	}
	return nil
}

// ============================================================
// 内部：公开输入重建 + 一致性校验
// ============================================================

// rebuildMatrix 将 V1 公开输入扁平数组 (row-major: i*N*T + j*T + t) 重建为 [N][N][T]
func rebuildMatrix(flat []float64, n int) ([][][]float64, error) {
	if n < 2 || len(flat) == 0 || len(flat)%(n*n) != 0 {
		return nil, fmt.Errorf("扁平数组长度 %d 无法按 N=%d 重建", len(flat), n)
	}
	t := len(flat) / (n * n)
	mat := make([][][]float64, n)
	for i := 0; i < n; i++ {
		mat[i] = make([][]float64, n)
		for j := 0; j < n; j++ {
			mat[i][j] = make([]float64, t)
			for k := 0; k < t; k++ {
				mat[i][j][k] = flat[i*n*t+j*t+k]
			}
		}
	}
	return mat, nil
}

// matricesEqual 逐元素比较两个 [N][N][T] 矩阵（浮点容差）
func matricesEqual(a, b [][][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if len(a[i][j]) != len(b[i][j]) {
				return false
			}
			for k := range a[i][j] {
				if math.Abs(a[i][j][k]-b[i][j][k]) > admmEpsilon {
					return false
				}
			}
		}
	}
	return true
}

// ============================================================
// 内部：聚合 settlement（链上派生）
// ============================================================

// aggregateSettlement 把 [N][N][T] 按 MG 对按时间聚合为净能量/净金额
// netEnergy[i][j] = Σ_t P_global[i][j][t] (kWh, 正值 = i 从 j 买入)
// netAmount[i][j] = Σ_t P_global[i][j][t] * p2p_price[i][j][t] (元, 反对称)
func aggregateSettlement(pGlobal, p2pPrice [][][]float64, n int) ([][]float64, [][]float64) {
	t := len(pGlobal[0][0])
	netEnergy := make([][]float64, n)
	netAmount := make([][]float64, n)
	for i := 0; i < n; i++ {
		netEnergy[i] = make([]float64, n)
		netAmount[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			var e, a float64
			for k := 0; k < t; k++ {
				e += pGlobal[i][j][k]
				a += pGlobal[i][j][k] * p2pPrice[i][j][k]
			}
			netEnergy[i][j] = e
			netAmount[i][j] = a
		}
	}
	return netEnergy, netAmount
}

// ============================================================
// 内部：账本结算（直接余额转账）
// ============================================================

// executeSettlement 按聚合结果执行余额转账
// netEnergy[i][j] > 0: i 从 j 买入 (i 付钱收能, j 收钱出能)
func (c *EnergyTradingContract) executeSettlement(
	ctx contractapi.TransactionContextInterface,
	microgridIDs []string,
	netEnergy, netAmount [][]float64,
	n int,
) error {
	// 先加载所有微网，检查存在性与资源充足
	mgs := make(map[string]*MicroGrid)
	for _, id := range microgridIDs {
		mg, err := c.GetMicroGrid(ctx, id)
		if err != nil {
			return fmt.Errorf("微网 %s 不存在: %w", id, err)
		}
		mgs[id] = mg
	}

	// 计算每个微网的净变化量（避免重复读写）
	type delta struct {
		energyWh int64 // 能源变化 (Wh, + 为收入)
		amountFen int64 // 现金变化 (分, + 为收入)
	}
	deltas := make(map[string]*delta)
	for _, id := range microgridIDs {
		deltas[id] = &delta{}
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			e := netEnergy[i][j] // >0: i 从 j 买入
			a := netAmount[i][j]
			if math.Abs(e) < admmEpsilon {
				continue
			}
			energyWh := int64(math.Round(e * 1000))   // kWh → Wh
			amountFen := int64(math.Round(a * 100))   // 元 → 分
			buyerID := microgridIDs[i]
			sellerID := microgridIDs[j]
			// buyer: 收能 (+energyWh), 付钱 (-amountFen)
			deltas[buyerID].energyWh += energyWh
			deltas[buyerID].amountFen -= amountFen
			// seller: 出能 (-energyWh), 收钱 (+amountFen)
			deltas[sellerID].energyWh -= energyWh
			deltas[sellerID].amountFen += amountFen
		}
	}

	// 校验资源充足并应用
	for id, d := range deltas {
		mg := mgs[id]
		if mg.Balance+d.amountFen < 0 {
			return fmt.Errorf("微网 %s 余额不足: 当前 %d 分, 需支付 %d 分", id, mg.Balance, -d.amountFen)
		}
		if mg.EnergyBalance+d.energyWh < 0 {
			return fmt.Errorf("微网 %s 能源不足: 当前 %d Wh, 需支出 %d Wh", id, mg.EnergyBalance, -d.energyWh)
		}
		mg.Balance += d.amountFen
		mg.EnergyBalance += d.energyWh
		mg.TotalTrades++
		mg.SuccessfulTrades++
		if mg.CreditScore < MaxCreditScore {
			mg.CreditScore++
		}
		mgKey, _ := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{id})
		mgJSON, _ := json.Marshal(mg)
		if err := ctx.GetStub().PutState(mgKey, mgJSON); err != nil {
			return fmt.Errorf("保存微网 %s 失败: %w", id, err)
		}
	}

	return nil
}

// ============================================================
// 内部：存储
// ============================================================

func (c *EnergyTradingContract) getADMMVK(ctx contractapi.TransactionContextInterface) (VerificationKey, error) {
	vkJSON, err := ctx.GetStub().GetState(admmVKKey)
	if err != nil {
		return VerificationKey{}, fmt.Errorf("读取 VK 失败: %w", err)
	}
	if vkJSON == nil {
		return VerificationKey{}, fmt.Errorf("验证密钥未设置，请先调用 SetADMMVerificationKey")
	}
	var vk VerificationKey
	if err := json.Unmarshal(vkJSON, &vk); err != nil {
		return VerificationKey{}, fmt.Errorf("解析 VK 失败: %w", err)
	}
	return vk, nil
}

func (c *EnergyTradingContract) getADMMSettlementRecord(ctx contractapi.TransactionContextInterface, sessionID string) (*ADMMSettlementRecord, error) {
	key, err := ctx.GetStub().CreateCompositeKey("ADMMSettlement", []string{sessionID})
	if err != nil {
		return nil, err
	}
	recJSON, err := ctx.GetStub().GetState(key)
	if err != nil {
		return nil, err
	}
	if recJSON == nil {
		return nil, nil
	}
	var rec ADMMSettlementRecord
	if err := json.Unmarshal(recJSON, &rec); err != nil {
		return nil, err
	}
	return &rec, nil
}

func (c *EnergyTradingContract) saveADMMSettlementRecord(ctx contractapi.TransactionContextInterface, rec ADMMSettlementRecord) error {
	key, err := ctx.GetStub().CreateCompositeKey("ADMMSettlement", []string{rec.SessionID})
	if err != nil {
		return err
	}
	recJSON, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, recJSON)
}
