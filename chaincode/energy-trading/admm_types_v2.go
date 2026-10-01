package main

import (
	"fmt"
	"math/big"
)

// ============================================================
// PublicInput V2 — per-MG 结算证明语句 (与 zkp_circuit/circuit_v2.go
// 的 MGStatementV2 逐字段对齐; 字段顺序自 V2 起冻结)
//
// 设计原则: 证明语句 = 结算契约。V2 证明的是现行市场方程
// (含 FiT 上网/柴油/η=0.95 SOC/互斥/全容量 RC/弃电上界/成本一致性),
// 每微网一份证明, 规模与系统 N 无关。
//
// 字段顺序 (FROZEN, 全部为整数域元素 ×1e6; cost_claim 为皮美元 ×1e12$):
//   [0]  session_hash   会话绑定 (sha256 低 43 位 ×1e6)
//   [1]  version_tag    电路/市场版本 (当前 2)
//   [2]  cost_claim     成本声明 = Σ(ToU·(购电+网损) + 燃料·DG
//                       + 磨损·放电 − FiT·上网), 皮美元
//   [3]  trade          [TradeSlotsV2][T] 逐对交易行: 按注册微网序去掉自身,
//                       非成员槽位补 0; 净进口 Σ_j P_ij 由电路内行和导出
//                       (2026-09-18 起 — 此行把双边分解并入证明语句)
//   [4]  grid_price     [T]   ToU
//   [5]  fit_price      [T]   FiT
//   [6]  loss_alloc     [T]   网损分摊
//   [7]  grid_cap       购电/上网上限
//   [8]  dg_cap         柴油容量
//   [9]  ess_power      储能功率上限
//   [10] eta_num        η 分子 (19)
//   [11] eta_den        η 分母 (20)
//   [12] s_min          [T+1] SOC 下带
//   [13] s_max          [T+1] SOC 上带
//   [14] dg_cost        柴油燃料成本 ($/kWh ×1e6)
//   [15] wear_cost      储能磨损成本 ($/kWh ×1e6)
//
// 公开输入域元素总数 = 10 + TradeSlotsV2·T + 3T + 2(T+1)
//   (T=24, TradeSlotsV2=19 时 588)
// 容量/η/带从"私密自声明"改为公开输入是 V2 的关键修复:
// v1 中 PMax 属于私密 witness, "≤ 自声明上限"的 range check 无证明力。
// 逐对交易行同为 V2 关键修复之二: 净进口此前是公开输入、行和只在链码明文层
// 校验, 留下一个保持各微网净进口不变而改变两两付款的"环流"自由度; 现在矩阵
// 的每个元素都进入交易双方各自的证明, 由链码反对称交叉核对钉死。
// ============================================================

const PublicInputVersionV2 = 2

// TradeSlotsV2 逐对交易行槽位数 = 注册市场规模上限 − 1
// (须与 zkp_circuit/circuit_v2.go 的 TradeSlotsV2 = MarketSizeMaxV2 − 1 一致;
// 语句形状由该常数冻结, 一个电路/一次 setup 覆盖所有联盟规模)
const TradeSlotsV2 = 19

// PublicInputV2 per-MG 证明的公开输入 (整数域元素, JSON 传输用 string 防 int64 溢出;
// cost_claim 理论上界 < 9e18 int64 内, 统一 string 最稳妥)
type PublicInputV2 struct {
	SessionHash string     `json:"sessionHash"` // [0]
	VersionTag  string     `json:"versionTag"`  // [1]
	CostClaim   string     `json:"costClaim"`   // [2]
	Trade       [][]string `json:"trade"`       // [3]  [TradeSlotsV2][T]
	GridPrice   []string   `json:"gridPrice"`   // [4]  [T]
	FitPrice    []string `json:"fitPrice"`    // [5]  [T]
	LossAlloc   []string `json:"lossAlloc"`   // [6]  [T]
	GridCap     string   `json:"gridCap"`     // [7]
	DgCap       string   `json:"dgCap"`       // [8]
	EssPower    string   `json:"essPower"`    // [9]
	EtaNum      string   `json:"etaNum"`      // [10]
	EtaDen      string   `json:"etaDen"`      // [11]
	SMin        []string `json:"sMin"`        // [12] [T+1]
	SMax        []string `json:"sMax"`        // [13] [T+1]
	DgCost      string   `json:"dgCost"`      // [14]
	WearCost    string   `json:"wearCost"`    // [15]
}

func bigFromString(s string) (*big.Int, error) {
	v, ok := new(big.Int).SetString(s, 10)
	if !ok {
		return nil, fmt.Errorf("非法域元素: %q", s)
	}
	return v, nil
}

// PublicInputV2ToFields 按 V2 冻结顺序组装域元素 (与 gnark 电路公开输入一致)
func PublicInputV2ToFields(pi PublicInputV2) ([]*big.Int, error) {
	one := func(s string) (*big.Int, error) { return bigFromString(s) }
	vec := func(xs []string) ([]*big.Int, error) {
		out := make([]*big.Int, len(xs))
		for i, x := range xs {
			v, err := one(x)
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	}
	var fields []*big.Int
	appendOne := func(s string) error {
		v, err := one(s)
		if err != nil {
			return err
		}
		fields = append(fields, v)
		return nil
	}
	appendVec := func(xs []string) error {
		vs, err := vec(xs)
		if err != nil {
			return err
		}
		fields = append(fields, vs...)
		return nil
	}
	for _, s := range []string{pi.SessionHash, pi.VersionTag, pi.CostClaim} {
		if err := appendOne(s); err != nil {
			return nil, err
		}
	}
	for _, row := range pi.Trade {
		if err := appendVec(row); err != nil {
			return nil, err
		}
	}
	if err := appendVec(pi.GridPrice); err != nil {
		return nil, err
	}
	if err := appendVec(pi.FitPrice); err != nil {
		return nil, err
	}
	if err := appendVec(pi.LossAlloc); err != nil {
		return nil, err
	}
	for _, s := range []string{pi.GridCap, pi.DgCap, pi.EssPower,
		pi.EtaNum, pi.EtaDen} {
		if err := appendOne(s); err != nil {
			return nil, err
		}
	}
	if err := appendVec(pi.SMin); err != nil {
		return nil, err
	}
	if err := appendVec(pi.SMax); err != nil {
		return nil, err
	}
	for _, s := range []string{pi.DgCost, pi.WearCost} {
		if err := appendOne(s); err != nil {
			return nil, err
		}
	}
	return fields, nil
}

// PublicInputV2FieldCount V2 公开输入域元素总数
// = 10 + TradeSlotsV2·T + 3T + 2(T+1)
// (逐对交易行为 TradeSlotsV2 个 T 长向量; 3 个 T 长向量:
//  grid_price/fit_price/loss_alloc; 2 个 T+1 向量: s_min/s_max; 10 个标量)
func PublicInputV2FieldCount(t int) int {
	return 10 + TradeSlotsV2*t + 3*t + 2*(t+1)
}

// ValidatePublicInputV2 结构校验: 版本号/向量长度/域内非负
func ValidatePublicInputV2(pi PublicInputV2) error {
	_, err := ValidateAndParsePublicInputV2(pi)
	return err
}

// ValidateAndParsePublicInputV2 一次遍历完成结构校验与域元素组装:
// 版本号 / 向量长度 / 域内非负, 并按冻结顺序返回域元素。
//
// 与 ValidatePublicInputV2 + PublicInputV2ToFields 串联调用语义相同, 但每个
// 域元素只解析一次 —— 此前验证与组装各解析一遍 (T=24 时 588 个字段共 1176 次
// 十进制字符串 → big.Int), 是单次证明验证耗时的主要来源之一。
func ValidateAndParsePublicInputV2(pi PublicInputV2) ([]*big.Int, error) {
	if len(pi.Trade) != TradeSlotsV2 {
		return nil, fmt.Errorf("V2: 交易行槽位数 %d ≠ TradeSlotsV2=%d",
			len(pi.Trade), TradeSlotsV2)
	}
	t := len(pi.Trade[0])
	for s, row := range pi.Trade {
		if len(row) != t {
			return nil, fmt.Errorf("V2: 交易行第 %d 槽长度 %d ≠ T=%d", s, len(row), t)
		}
	}
	if len(pi.GridPrice) != t || len(pi.FitPrice) != t || len(pi.LossAlloc) != t {
		return nil, fmt.Errorf("V2: 价格/网损向量长度应为 T=%d", t)
	}
	if len(pi.SMin) != t+1 || len(pi.SMax) != t+1 {
		return nil, fmt.Errorf("V2: SOC 带长度应为 T+1=%d", t+1)
	}

	fields := make([]*big.Int, 0, PublicInputV2FieldCount(t))
	addOne := func(s string) error {
		v, err := bigFromString(s)
		if err != nil {
			return err
		}
		if v.Sign() < 0 {
			return fmt.Errorf("V2: 公开输入出现负域元素 %s", s)
		}
		fields = append(fields, v)
		return nil
	}
	addVec := func(xs []string) error {
		for _, s := range xs {
			if err := addOne(s); err != nil {
				return err
			}
		}
		return nil
	}

	// 顺序必须与 MGStatementV2 的字段声明逐项一致 (冻结于 V2)
	if err := addOne(pi.SessionHash); err != nil {
		return nil, err
	}
	if err := addOne(pi.VersionTag); err != nil {
		return nil, err
	}
	if fields[1].Int64() != PublicInputVersionV2 {
		return nil, fmt.Errorf("V2: version_tag=%s ≠ %d (跨版本重放拒绝)",
			pi.VersionTag, PublicInputVersionV2)
	}
	if err := addOne(pi.CostClaim); err != nil {
		return nil, err
	}
	for _, row := range pi.Trade {
		if err := addVec(row); err != nil {
			return nil, err
		}
	}
	for _, xs := range [][]string{pi.GridPrice, pi.FitPrice, pi.LossAlloc} {
		if err := addVec(xs); err != nil {
			return nil, err
		}
	}
	for _, s := range []string{pi.GridCap, pi.DgCap, pi.EssPower,
		pi.EtaNum, pi.EtaDen} {
		if err := addOne(s); err != nil {
			return nil, err
		}
	}
	for _, xs := range [][]string{pi.SMin, pi.SMax} {
		if err := addVec(xs); err != nil {
			return nil, err
		}
	}
	for _, s := range []string{pi.DgCost, pi.WearCost} {
		if err := addOne(s); err != nil {
			return nil, err
		}
	}

	if want := PublicInputV2FieldCount(t); len(fields) != want {
		return nil, fmt.Errorf("V2: 公开输入域元素数 %d ≠ %d", len(fields), want)
	}
	return fields, nil
}

func concat(vs ...[]string) []string {
	out := []string{}
	for _, v := range vs {
		out = append(out, v...)
	}
	return out
}

// CheckLineCapacity 线路容量校验 (V2 新增, 链码明文执行):
// 交易是公开输入, 无需进电路 — 逐对逐槽校验 |P_ij| ≤ L_ij.
// trades row-major [N*N*T], caps row-major [N*(N-1)] (i<j 顺序, i*(2K-i-1)/2+j 映射)
func CheckLineCapacity(trades []float64, caps []float64, n, t int) error {
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			k := i*(2*n-i-1)/2 + (j - i - 1)
			if k >= len(caps) {
				return fmt.Errorf("线容量索引越界 k=%d", k)
			}
			for s := 0; s < t; s++ {
				p := trades[i*n*t+j*t+s]
				if p > caps[k] || p < -caps[k] {
					return fmt.Errorf(
						"线路容量违规 MG%d→MG%d slot%d: |%.4f| > %.4f kW",
						i+1, j+1, s, p, caps[k])
				}
			}
		}
	}
	return nil
}
