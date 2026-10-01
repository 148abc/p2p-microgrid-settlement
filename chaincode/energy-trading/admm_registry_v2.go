package main

// ============================================================
// V2 市场配置注册表 —— 证明语句参数的链上锚定
//
// 背景 (论文 §V-A 的配置边界): V2 电路的容量上限、SOC 带、η、费率与网损
// 分摊是"公开输入", 但它们随结算载荷由提交方传入。公开只有在账本独立固定
// 这些值时才等于锚定; 否则受损网关可以自声明配置 (例如 η=1/1 或放宽 SOC
// 带) 并仍然给出可接受的证明。
//
// 本文件把每会话的市场配置在开市时固定到链上 (管理员写, 与线路容量表同规),
// 结算时逐微网比对公开输入与注册值, 并由注册费率与类型折扣在合约内重算中间
// 价矩阵 π_ij^t = ½(α_i+α_j)·π^t。缺注册表、任一字段偏离、或价格矩阵不符,
// 结算一律被拒绝。
// ============================================================

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// admmMarketConfigV2Prefix 链上市场配置键前缀 (按 sessionID 分键)
const admmMarketConfigV2Prefix = "ADMM_V2_MARKET_CONFIG_"

// admmMidPriceTolV2 中间价比对容差 (域单元, 价格分辨率 1e-6 $/kWh)。
// 取两个域单元, 吸收提交方的舍入方向差异; 更粗的偏差即视为篡改。
const admmMidPriceTolV2 = 2e-6

// admmBandTolV2 SOC 带的锚定容差 (域单元; 100 = 1e-4 kWh)。
// 整数 SOC 格点由量化后的充放电递推得出, 与市场带的差可达数十个域单元
// (实测 ≤ 24), 因此 SOC 带按"注册值 ± 该容差"锚定; 容差比带宽小 6 个数量级,
// 对储能约束没有实际松弛空间, 而其余参数 (容量/η/费率/成本/网损) 仍逐字段精确相等。
const admmBandTolV2 = int64(100)

// ADMMMarketMGConfigV2 单微网的注册配置。字段与 PublicInputV2 的配置字段
// 逐项同名同口径 (域元素十进制字符串), 以便结算时做逐字段等值比对。
type ADMMMarketMGConfigV2 struct {
	GridCap   string   `json:"gridCap"`
	DgCap     string   `json:"dgCap"`
	EssPower  string   `json:"essPower"`
	EtaNum    string   `json:"etaNum"`
	EtaDen    string   `json:"etaDen"`
	DgCost    string   `json:"dgCost"`
	WearCost  string   `json:"wearCost"`
	SMin      []string `json:"sMin"`
	SMax      []string `json:"sMax"`
	GridPrice []string `json:"gridPrice"`
	FitPrice  []string `json:"fitPrice"`
	LossAlloc []string `json:"lossAlloc"`
}

// ADMMMarketConfigV2 一次会话的链上市场配置
type ADMMMarketConfigV2 struct {
	SessionID    string                 `json:"sessionID"`
	MicrogridIDs []string               `json:"microgridIDs"`
	T            int                    `json:"t"`
	Alpha        []string               `json:"alpha"` // 逐微网类型折扣 (十进制, 0<α<1)
	Microgrids   []ADMMMarketMGConfigV2 `json:"microgrids"`
}

// ------------------------------------------------------------
// 注册与查询
// ------------------------------------------------------------

// RegisterMarketConfigV2 开市时在链上固定本会话的市场配置 (仅管理员)。
// 与 SetLineCapacities 同规: 缺此表则 V2 结算被拒绝。
func (c *EnergyTradingContract) RegisterMarketConfigV2(
	ctx contractapi.TransactionContextInterface,
	payloadJSON string,
) error {
	if err := requireAdmin(ctx); err != nil {
		return err
	}
	var cfg ADMMMarketConfigV2
	if err := json.Unmarshal([]byte(payloadJSON), &cfg); err != nil {
		return fmt.Errorf("解析市场配置失败: %w", err)
	}
	if err := cfg.validate(); err != nil {
		return err
	}
	b, err := json.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("序列化市场配置失败: %w", err)
	}
	if err := ctx.GetStub().PutState(admmMarketConfigV2Prefix+cfg.SessionID, b); err != nil {
		return fmt.Errorf("写入市场配置失败: %w", err)
	}
	ctx.GetStub().SetEvent("RegisterMarketConfigV2", []byte(cfg.SessionID))
	return nil
}

// GetMarketConfigV2 查询链上注册的市场配置 (明文只读)
func (c *EnergyTradingContract) GetMarketConfigV2(
	ctx contractapi.TransactionContextInterface,
	sessionID string,
) (string, error) {
	b, err := ctx.GetStub().GetState(admmMarketConfigV2Prefix + sessionID)
	if err != nil {
		return "", fmt.Errorf("读取市场配置失败: %w", err)
	}
	if b == nil {
		return "", fmt.Errorf("会话 %s 的市场配置未注册", sessionID)
	}
	return string(b), nil
}

func (c *EnergyTradingContract) getMarketConfigV2(
	ctx contractapi.TransactionContextInterface,
	sessionID string,
) (*ADMMMarketConfigV2, error) {
	s, err := c.GetMarketConfigV2(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("%w: 结算要求语句参数已锚定", err)
	}
	var cfg ADMMMarketConfigV2
	if err := json.Unmarshal([]byte(s), &cfg); err != nil {
		return nil, fmt.Errorf("市场配置反序列化失败: %w", err)
	}
	return &cfg, nil
}

// ------------------------------------------------------------
// 结构校验
// ------------------------------------------------------------

func (cfg *ADMMMarketConfigV2) validate() error {
	n := len(cfg.MicrogridIDs)
	if n < 2 {
		return fmt.Errorf("市场配置: 微网数量 %d < 2", n)
	}
	if cfg.SessionID == "" {
		return fmt.Errorf("市场配置: sessionID 为空")
	}
	if cfg.T < 1 {
		return fmt.Errorf("市场配置: T=%d 非法", cfg.T)
	}
	if len(cfg.Alpha) != n || len(cfg.Microgrids) != n {
		return fmt.Errorf("市场配置: alpha/microgrids 长度与 N=%d 不符 (%d/%d)",
			n, len(cfg.Alpha), len(cfg.Microgrids))
	}
	for i, a := range cfg.Alpha {
		v, ok := new(big.Float).SetString(a)
		if !ok {
			return fmt.Errorf("市场配置: alpha[%d]=%q 非法", i, a)
		}
		f, _ := v.Float64()
		if !(f > 0 && f < 1) {
			return fmt.Errorf("市场配置: alpha[%d]=%v 须在 (0,1)", i, f)
		}
	}
	for i := range cfg.Microgrids {
		if err := cfg.Microgrids[i].validate(cfg.T); err != nil {
			return fmt.Errorf("市场配置 MG%d: %w", i+1, err)
		}
	}
	return nil
}

func (m *ADMMMarketMGConfigV2) validate(t int) error {
	scalars := []struct {
		name string
		val  string
		pos  bool
	}{
		{"gridCap", m.GridCap, true},
		{"dgCap", m.DgCap, false},
		{"essPower", m.EssPower, false},
		{"etaNum", m.EtaNum, true},
		{"etaDen", m.EtaDen, true},
		{"dgCost", m.DgCost, false},
		{"wearCost", m.WearCost, false},
	}
	for _, s := range scalars {
		v, err := bigFromString(s.val)
		if err != nil {
			return fmt.Errorf("%s 解析失败: %w", s.name, err)
		}
		if s.pos && v.Sign() <= 0 {
			return fmt.Errorf("%s 须为正", s.name)
		}
	}
	vecs := []struct {
		name string
		val  []string
		want int
	}{
		{"sMin", m.SMin, t + 1},
		{"sMax", m.SMax, t + 1},
		{"gridPrice", m.GridPrice, t},
		{"fitPrice", m.FitPrice, t},
		{"lossAlloc", m.LossAlloc, t},
	}
	for _, v := range vecs {
		if len(v.val) != v.want {
			return fmt.Errorf("%s 长度 %d != %d", v.name, len(v.val), v.want)
		}
		for k, s := range v.val {
			if _, err := bigFromString(s); err != nil {
				return fmt.Errorf("%s[%d] 解析失败: %w", v.name, k, err)
			}
		}
	}
	return nil
}

// ------------------------------------------------------------
// 结算时的锚定检查
// ------------------------------------------------------------

// sameField 两个十进制域元素是否等值 (按大整数比较, 容忍前导零等表层差异)
func sameField(a, b string) (bool, error) {
	x, err := bigFromString(a)
	if err != nil {
		return false, err
	}
	y, err := bigFromString(b)
	if err != nil {
		return false, err
	}
	return x.Cmp(y) == 0, nil
}

func anchorScalarV2(name, got, want string, mgIdx int) error {
	ok, err := sameField(got, want)
	if err != nil {
		return fmt.Errorf("MG%d %s 解析失败: %w", mgIdx+1, name, err)
	}
	if !ok {
		return fmt.Errorf(
			"MG%d %s 与链上注册配置不符 (证明 %s, 注册 %s): 结算被拒绝",
			mgIdx+1, name, got, want)
	}
	return nil
}

func anchorVecV2(name string, got, want []string, mgIdx int) error {
	if len(got) != len(want) {
		return fmt.Errorf("MG%d %s 长度 %d 与注册配置 %d 不符", mgIdx+1, name, len(got), len(want))
	}
	for k := range got {
		ok, err := sameField(got[k], want[k])
		if err != nil {
			return fmt.Errorf("MG%d %s[%d] 解析失败: %w", mgIdx+1, name, k, err)
		}
		if !ok {
			return fmt.Errorf(
				"MG%d %s[%d] 与链上注册配置不符 (证明 %s, 注册 %s): 结算被拒绝",
				mgIdx+1, name, k, got[k], want[k])
		}
	}
	return nil
}

// anchorBandV2 SOC 带锚定: 允许 |提交值 - 注册值| <= admmBandTolV2 个域单元
// (量化裕度), 超出即拒绝。带下界不得低于注册值减容差, 上界不得高于注册值加容差,
// 因此不存在"用自生成包络替换市场带"的空间。
func anchorBandV2(name string, got, want []string, mgIdx int) error {
	if len(got) != len(want) {
		return fmt.Errorf("MG%d %s 长度 %d 与注册配置 %d 不符", mgIdx+1, name, len(got), len(want))
	}
	tol := big.NewInt(admmBandTolV2)
	for k := range got {
		x, err := bigFromString(got[k])
		if err != nil {
			return fmt.Errorf("MG%d %s[%d] 解析失败: %w", mgIdx+1, name, k, err)
		}
		y, err := bigFromString(want[k])
		if err != nil {
			return fmt.Errorf("MG%d %s[%d] 注册值解析失败: %w", mgIdx+1, name, k, err)
		}
		diff := new(big.Int).Sub(x, y)
		diff.Abs(diff)
		if diff.Cmp(tol) > 0 {
			return fmt.Errorf(
				"MG%d %s[%d] 偏离链上注册配置 %s 个域单元 (容差 %d): 结算被拒绝",
				mgIdx+1, name, k, diff.String(), admmBandTolV2)
		}
	}
	return nil
}

// anchorPayloadToConfigV2 逐微网比对公开输入与注册配置, 并按注册费率与类型
// 折扣重算中间价矩阵。任一项偏离即拒绝, 且不会发生任何状态变更。
func anchorPayloadToConfigV2(cfg *ADMMMarketConfigV2, req *SettleADMMV2Request) error {
	n := len(req.MicrogridIDs)
	if len(cfg.MicrogridIDs) != n {
		return fmt.Errorf("结算载荷 N=%d 与注册配置 N=%d 不符", n, len(cfg.MicrogridIDs))
	}
	for i := 0; i < n; i++ {
		if req.MicrogridIDs[i] != cfg.MicrogridIDs[i] {
			return fmt.Errorf("结算载荷 MG%d=%s 与注册配置 %s 不符",
				i+1, req.MicrogridIDs[i], cfg.MicrogridIDs[i])
		}
	}
	for i := 0; i < n; i++ {
		pi := req.PublicInputs[i]
		reg := cfg.Microgrids[i]
		if len(pi.GridPrice) != cfg.T {
			return fmt.Errorf("MG%d 公开输入 T=%d 与注册配置 T=%d 不符",
				i+1, len(pi.GridPrice), cfg.T)
		}
		for _, s := range []struct{ name, got, want string }{
			{"grid_cap", pi.GridCap, reg.GridCap},
			{"dg_cap", pi.DgCap, reg.DgCap},
			{"ess_power", pi.EssPower, reg.EssPower},
			{"eta_num", pi.EtaNum, reg.EtaNum},
			{"eta_den", pi.EtaDen, reg.EtaDen},
			{"dg_cost", pi.DgCost, reg.DgCost},
			{"wear_cost", pi.WearCost, reg.WearCost},
		} {
			if err := anchorScalarV2(s.name, s.got, s.want, i); err != nil {
				return err
			}
		}
		for _, v := range []struct {
			name string
			got  []string
			want []string
		}{
			{"soc_band_min", pi.SMin, reg.SMin},
			{"soc_band_max", pi.SMax, reg.SMax},
		} {
			if err := anchorBandV2(v.name, v.got, v.want, i); err != nil {
				return err
			}
		}
		for _, v := range []struct {
			name string
			got  []string
			want []string
		}{
			{"grid_price", pi.GridPrice, reg.GridPrice},
			{"fit_price", pi.FitPrice, reg.FitPrice},
			{"loss_alloc", pi.LossAlloc, reg.LossAlloc},
		} {
			if err := anchorVecV2(v.name, v.got, v.want, i); err != nil {
				return err
			}
		}
	}
	return checkMidPriceMatrixV2(cfg, req)
}

// checkMidPriceMatrixV2 合约内重算结算中间价:
//
//	π_ij^t = ½(α_i + α_j) · π^t
//
// 注册的 π^t 为 ToU 向量, α_i 为微网类型折扣 (论文式 (3))。提交的 P2P 价格
// 矩阵必须逐槽等于重算值 (容差 admmMidPriceTolV2), 否则拒绝。这条检查把
// "定价规则" 从提交方声明变为账本强制。
func checkMidPriceMatrixV2(cfg *ADMMMarketConfigV2, req *SettleADMMV2Request) error {
	n := len(req.MicrogridIDs)
	if len(req.P2PPrice) != n {
		return fmt.Errorf("价格矩阵 N=%d 与结算载荷 N=%d 不符", len(req.P2PPrice), n)
	}
	alpha := make([]float64, n)
	for i := 0; i < n; i++ {
		v, ok := new(big.Float).SetString(cfg.Alpha[i])
		if !ok {
			return fmt.Errorf("注册配置 alpha[%d]=%q 非法", i, cfg.Alpha[i])
		}
		alpha[i], _ = v.Float64()
	}
	// 注册的 ToU 向量 (域元素 -> $/kWh)
	tou := make([]float64, cfg.T)
	for k := 0; k < cfg.T; k++ {
		v, err := bigFromString(cfg.Microgrids[0].GridPrice[k])
		if err != nil {
			return fmt.Errorf("注册 ToU[%d] 解析失败: %w", k, err)
		}
		f := new(big.Float).SetInt(v)
		f.Quo(f, big.NewFloat(1e6))
		tou[k], _ = f.Float64()
	}
	for i := 0; i < n; i++ {
		if len(req.P2PPrice[i]) != n {
			return fmt.Errorf("价格矩阵行 %d 维度 %d != N=%d", i+1, len(req.P2PPrice[i]), n)
		}
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			if len(req.P2PPrice[i][j]) != cfg.T {
				return fmt.Errorf("价格矩阵[%d][%d] 长度 %d != T=%d",
					i+1, j+1, len(req.P2PPrice[i][j]), cfg.T)
			}
			coef := 0.5 * (alpha[i] + alpha[j])
			for k := 0; k < cfg.T; k++ {
				want := coef * tou[k]
				got := req.P2PPrice[i][j][k]
				if math.IsNaN(got) || math.Abs(got-want) > admmMidPriceTolV2 {
					return fmt.Errorf(
						"MG%d-MG%d slot%d 结算价 %.9f 与合约重算的中间价 %.9f 不符 "+
							"(α=%.3f, ToU=%.6f): 定价规则未被遵守, 结算被拒绝",
						i+1, j+1, k, got, want, coef, tou[k])
				}
			}
		}
	}
	return nil
}
