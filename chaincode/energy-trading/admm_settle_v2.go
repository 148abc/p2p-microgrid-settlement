package main

// ============================================================
// SettleADMMV2 — 部署路径的 per-microgrid 结算
//
// 与 SettleADMM (V1) 的差别在"谁出证明":
//   V1 语句是 fleet 级: 一份证明覆盖全队, 公开输入含完整 [N][N][T] 交易矩阵
//   与价格矩阵, 因此出证明的一方必须掌握全体微网的终态 —— 集中式。
//   V2 语句是 per-microgrid: 每微网一份证明, 公开输入 24T+12 个域元素
//   (T=24: 588), 覆盖本微网的有向交易行与成本声明; 出证明的一方只需本微网
//   的私有信息, witness 不离开微网。这与论文 §V-D 的 primary deployment mode 一致。
//
// 代价与边界 (2026-09-18 更新): V2 语句现已包含逐对交易行, 因此结算依据的
// 交易矩阵由本合约按下述链条约束:
//   (a) 反驳性: 每个元素进入交易双方各自的证明语句 (公开输入 trade 行),
//       净进口 Σ_j P_ij 由电路内行和导出 —— 矩阵元素不再是"明文声明"
//   (b) 反对称   P_ij = -P_ji                      (validateADMMFinalState)
//   (c) 交叉核对 载荷矩阵 == 各微网被证明的交易行   (checkTradeRowAgainstMatrix)
//   (d) 线路容量 |P_ij| <= L_ij                    (链上固定的容量表)
//   (e) 语句参数锚定 + 中间价重算                  (admm_registry_v2.go)
// (a)+(c) 使矩阵的每个元素都被对应双方的证明与合约核对共同钉死: 任何单边改动
// 都会撞上另一方的公开行或反对称, 此前"保持各微网净进口不变而搬运两两付款"的
// 环流自由度因此消失。(e) 仍把"价格"这一半闭合: 结算价必须等于链上注册费率与
// 类型折扣重算出的中间价 ½(α_i+α_j)π^t。
// ============================================================

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// admmLineCapV2Key 链上固定的线路容量表
const admmLineCapV2Key = "ADMM_V2_LINE_CAPACITY"

// SettleADMMV2Request 一次 V2 结算交易的完整载荷。
// 一份交易携带 N 份 (proof, publicInput) 对, 保证结算原子性;
// 载荷中不含任何私密量, 证明本身不泄露 witness。
type SettleADMMV2Request struct {
	SessionID    string          `json:"sessionID"`
	MicrogridIDs []string        `json:"microgridIDs"`
	PGlobal      [][][]float64   `json:"pGlobal"`
	P2PPrice     [][][]float64   `json:"p2pPrice"`
	Proofs       []ZKPProof      `json:"proofs"`
	PublicInputs []PublicInputV2 `json:"publicInputs"`
}

// ------------------------------------------------------------
// 线路容量表 (管理员在开市时固定; 缺此表则容量筛无意义)
// ------------------------------------------------------------

// SetLineCapacities 链上固定线路容量表 [N*(N-1) 条, i<j 打包顺序]
func (c *EnergyTradingContract) SetLineCapacities(
	ctx contractapi.TransactionContextInterface,
	capsJSON string,
) error {
	if err := requireAdmin(ctx); err != nil {
		return err
	}
	var caps []float64
	if err := json.Unmarshal([]byte(capsJSON), &caps); err != nil {
		return fmt.Errorf("解析线路容量失败: %w", err)
	}
	if len(caps) == 0 {
		return fmt.Errorf("线路容量表为空")
	}
	for i, v := range caps {
		if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
			return fmt.Errorf("线路容量[%d]=%v 非法 (须为正有限值)", i, v)
		}
	}
	b, err := json.Marshal(caps)
	if err != nil {
		return fmt.Errorf("序列化线路容量失败: %w", err)
	}
	return ctx.GetStub().PutState(admmLineCapV2Key, b)
}

// GetLineCapacities 查询链上固定的线路容量表
func (c *EnergyTradingContract) GetLineCapacities(
	ctx contractapi.TransactionContextInterface,
) (string, error) {
	b, err := ctx.GetStub().GetState(admmLineCapV2Key)
	if err != nil {
		return "", fmt.Errorf("读取线路容量失败: %w", err)
	}
	if b == nil {
		return "", fmt.Errorf("线路容量未设置")
	}
	return string(b), nil
}

func (c *EnergyTradingContract) getLineCapacities(
	ctx contractapi.TransactionContextInterface,
) ([]float64, error) {
	s, err := c.GetLineCapacities(ctx)
	if err != nil {
		return nil, err
	}
	var caps []float64
	if err := json.Unmarshal([]byte(s), &caps); err != nil {
		return nil, fmt.Errorf("线路容量反序列化失败: %w", err)
	}
	return caps, nil
}

// ------------------------------------------------------------
// 域元素 ↔ 带符号整数
// ------------------------------------------------------------

// fieldToSigned 把公开输入的域元素按补码约定反解为带符号整数。
// 电路把负数 (净卖出) 表示为 r-|x|, 链码的 ValidatePublicInputV2 同样要求
// 非负域元素, 所以做数值比较前必须反解。
func fieldToSigned(v *big.Int) *big.Int {
	mod := ecc.BN254.ScalarField()
	half := new(big.Int).Rsh(mod, 1)
	if v.Cmp(half) > 0 {
		return new(big.Int).Sub(v, mod)
	}
	return new(big.Int).Set(v)
}

// checkTradeRowAgainstMatrix 公开交易行与载荷矩阵的交叉核对。
// 槽位映射: 成员 j≠i ↔ 槽 (j<i ? j : j−1), 即按注册序去掉自身;
// 槽位 ≥ N−1 (非成员) 必须为 0。核对把矩阵的每个元素钉进对应双方的
// 证明语句 —— 与载荷矩阵的反对称校验叠加后, 环流自由度消失。
func checkTradeRowAgainstMatrix(mgIdx int, row [][]float64,
	tradePublic [][]string, n, t int) error {
	if len(tradePublic) != TradeSlotsV2 {
		return fmt.Errorf("MG%d 公开交易行槽位数 %d != %d",
			mgIdx+1, len(tradePublic), TradeSlotsV2)
	}
	if n > TradeSlotsV2+1 {
		return fmt.Errorf("结算规模 N=%d 超过语句定长市场规模 %d",
			n, TradeSlotsV2+1)
	}
	if len(row) != n {
		return fmt.Errorf("MG%d 交易矩阵行维度 %d != N=%d", mgIdx+1, len(row), n)
	}
	for slot := 0; slot < TradeSlotsV2; slot++ {
		if len(tradePublic[slot]) != t {
			return fmt.Errorf("MG%d 公开交易行第 %d 槽长度 %d != T=%d",
				mgIdx+1, slot, len(tradePublic[slot]), t)
		}
		for k := 0; k < t; k++ {
			want := new(big.Int) // 非成员槽位: 期望恒为 0
			if slot < n-1 {
				j := slot
				if slot >= mgIdx {
					j = slot + 1
				}
				if len(row[j]) != t {
					return fmt.Errorf("MG%d 矩阵列维度异常", mgIdx+1)
				}
				want = fieldToSigned(FloatToField(row[j][k]))
			}
			got, err := bigFromString(tradePublic[slot][k])
			if err != nil {
				return fmt.Errorf("MG%d 公开交易行[%d][%d] 解析失败: %w",
					mgIdx+1, slot, k, err)
			}
			if got = fieldToSigned(got); got.Cmp(want) != 0 {
				return fmt.Errorf(
					"MG%d 槽%d slot%d 交易未被证明绑定: 公开行 %s != 载荷矩阵 %s (域单元)",
					mgIdx+1, slot, k, got.String(), want.String())
			}
		}
	}
	return nil
}

// ------------------------------------------------------------
// 结算交易
// ------------------------------------------------------------

// settleV2Timing V2 结算耗时拆分 (毫秒)。
// 注意: 链码执行耗时只能经日志/事件输出, 不得写入账本状态 —— 各 peer
// 执行耗时不同会导致写集不一致, 交易被 vscc 判无效 (与 V1 同规)。
type settleV2Timing struct {
	JSONParse, Structure, Replay, LineCapacity, VKLoad float64
	Verify, Link, Aggregate, Settle, Record, Total     float64
}

func (tm settleV2Timing) String(sessionID string, n, tSlots, proofs int) string {
	return fmt.Sprintf(
		"SETTLE_V2_TIMING session=%s n=%d t=%d proofs=%d json=%.3f structure=%.3f "+
			"replay=%.3f capacity=%.3f vk_load=%.3f verify=%.3f link=%.3f "+
			"aggregate=%.3f settle=%.3f record=%.3f total=%.3f ms",
		sessionID, n, tSlots, proofs,
		tm.JSONParse, tm.Structure, tm.Replay, tm.LineCapacity, tm.VKLoad,
		tm.Verify, tm.Link, tm.Aggregate, tm.Settle, tm.Record, tm.Total)
}

// SettleADMMV2 per-microgrid 结算: 一份交易携带 N 份 V2 证明, 逐份验证通过后
// 才记账。任一份证明无效/会话不符/行和不匹配, 整个交易 abort, 无状态变更。
func (c *EnergyTradingContract) SettleADMMV2(
	ctx contractapi.TransactionContextInterface,
	payloadJSON string,
) error {
	var tm settleV2Timing
	t0 := time.Now()
	ms := func(s time.Time) float64 { return float64(time.Since(s).Microseconds()) / 1000.0 }

	s := time.Now()
	var req SettleADMMV2Request
	if err := json.Unmarshal([]byte(payloadJSON), &req); err != nil {
		return fmt.Errorf("解析 V2 结算载荷失败: %w", err)
	}
	tm.JSONParse = ms(s)

	n := len(req.MicrogridIDs)
	if n < 2 {
		return fmt.Errorf("微网数量不足: 需要 >=2, 实际 %d", n)
	}
	if len(req.Proofs) != n || len(req.PublicInputs) != n {
		return fmt.Errorf("证明/公开输入数量必须各为 %d, 实际 %d/%d",
			n, len(req.Proofs), len(req.PublicInputs))
	}

	// ---- 1. 维度 / 价格非负 / 反对称 ----
	s = time.Now()
	if err := validateADMMFinalState(req.MicrogridIDs, req.PGlobal, req.P2PPrice); err != nil {
		return err
	}
	tm.Structure = ms(s)
	t := len(req.PGlobal[0][0])

	// ---- 2. 重放保护 ----
	s = time.Now()
	if existing, _ := c.getADMMSettlementRecord(ctx, req.SessionID); existing != nil {
		return fmt.Errorf("sessionID %s 已结算，禁止重放", req.SessionID)
	}
	tm.Replay = ms(s)

	// ---- 2b. 市场配置锚定: 语句参数必须等于链上注册值, 中间价由合约重算 ----
	// 缺注册表即拒绝 (与线路容量表同规)。这一步把"公开输入"从调用方自声明
	// 变为账本固定值: 容量上限/SOC 带/η/费率/网损分摊任一偏离都会被拒绝。
	s = time.Now()
	cfg, err := c.getMarketConfigV2(ctx, req.SessionID)
	if err != nil {
		return err
	}
	if err := anchorPayloadToConfigV2(cfg, &req); err != nil {
		return err
	}
	tm.Structure += ms(s)

	// ---- 3. 线路容量 (链上固定表, 非调用方传入) ----
	s = time.Now()
	caps, err := c.getLineCapacities(ctx)
	if err != nil {
		return err
	}
	flat := make([]float64, 0, n*n*t)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			flat = append(flat, req.PGlobal[i][j]...)
		}
	}
	if err := CheckLineCapacity(flat, caps, n, t); err != nil {
		return err
	}
	tm.LineCapacity = ms(s)

	// ---- 4. 会话绑定 + 逐微网证明验证 + 交易行交叉核对 ----
	s = time.Now()
	vk, err := c.getADMMVK(ctx)
	if err != nil {
		return fmt.Errorf("加载验证密钥失败: %w", err)
	}
	// 验证器只构建一次, 全部 N 份证明复用: 构建要解析 VK 的全部曲线点
	// (IC = 公开输入数+1), 逐证重建会把这一项乘以 N
	verifier, err := NewGroth16Verifier(vk)
	if err != nil {
		return fmt.Errorf("创建验证器失败: %w", err)
	}
	tm.VKLoad = ms(s)
	expectedHash := FloatToField(SessionHashField(req.SessionID))
	for i := 0; i < n; i++ {
		pi := req.PublicInputs[i]

		// 会话绑定: 每份证明都必须绑定到本会话 (拒绝跨会话/跨微网搬运)
		got, err := bigFromString(pi.SessionHash)
		if err != nil {
			return fmt.Errorf("MG%d 会话哈希解析失败: %w", i+1, err)
		}
		if got.Cmp(expectedHash) != 0 {
			return fmt.Errorf(
				"MG%d 会话哈希与 sessionID 不一致 (证明绑定 %s, 本会话 %s): 重放被拒绝",
				i+1, got.String(), expectedHash.String())
		}

		// 公开输入维度必须与矩阵口径一致
		if gotT := len(pi.GridPrice); gotT != t {
			return fmt.Errorf("MG%d 公开输入 T=%d 与结算矩阵 T=%d 不一致", i+1, gotT, t)
		}

		sv := time.Now()
		ok, err := VerifyADMMProofV2With(verifier, req.Proofs[i], pi)
		if err != nil {
			return fmt.Errorf("MG%d V2 证明验证出错: %w", i+1, err)
		}
		if !ok {
			return fmt.Errorf("MG%d V2 证明无效，结算被拒绝", i+1)
		}
		tm.Verify += ms(sv)

		sl := time.Now()
		if err := checkTradeRowAgainstMatrix(i, req.PGlobal[i], pi.Trade, n, t); err != nil {
			return err
		}
		tm.Link += ms(sl)
	}

	// ---- 5. 账本结算 (与 V1 共用聚合与划转) ----
	s = time.Now()
	netEnergy, netAmount := aggregateSettlement(req.PGlobal, req.P2PPrice, n)
	tm.Aggregate = ms(s)

	s = time.Now()
	if err := c.executeSettlement(ctx, req.MicrogridIDs, netEnergy, netAmount, n); err != nil {
		return fmt.Errorf("账本结算失败: %w", err)
	}
	tm.Settle = ms(s)

	s = time.Now()
	txTimestamp, _ := ctx.GetStub().GetTxTimestamp()
	txID := ctx.GetStub().GetTxID()
	record := ADMMSettlementRecord{
		SessionID:      req.SessionID,
		MicroGridIDs:   req.MicrogridIDs,
		N:              n,
		T:              t,
		Converged:      true,
		NetEnergyKWh:   netEnergy,
		NetAmountYuan:  netAmount,
		ZKPVerified:    true,
		ConfigAnchored: true,
		SettledAt:      time.Unix(txTimestamp.Seconds, int64(txTimestamp.Nanos)),
		TxID:           txID,
	}
	if err := c.saveADMMSettlementRecord(ctx, record); err != nil {
		return fmt.Errorf("保存结算记录失败: %w", err)
	}
	tm.Record = ms(s)
	tm.Total = ms(t0)

	fmt.Println(tm.String(req.SessionID, n, t, len(req.Proofs)))

	eventPayload, _ := json.Marshal(map[string]interface{}{
		"sessionId":   req.SessionID,
		"n":           n,
		"t":           t,
		"mode":        "per-microgrid-v2",
		"proofCount":  len(req.Proofs),
		"zkpVerified": true,
		"txId":        txID,
	})
	ctx.GetStub().SetEvent("SettleADMMV2", eventPayload)

	return nil
}
