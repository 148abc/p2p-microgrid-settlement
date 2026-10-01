package main

import (
	"strconv"
	"testing"
)

// ============================================================
// V2 公开输入契约测试: 字段顺序/结构校验/线容量检查
// ============================================================

func samplePublicInputV2(t int) PublicInputV2 {
	s := func(v int64) string {
		return strconv.FormatInt(v, 10)
	}
	pi := PublicInputV2{
		SessionHash: s(1234500000),
		VersionTag:  s(2),
		CostClaim:   s(667950485375000),
		GridCap:     s(500000000),
		DgCap:       s(200000000),
		EssPower:    s(300000000),
		EtaNum:      s(19),
		EtaDen:      s(20),
		DgCost:      s(150000),
		WearCost:    s(15000),
	}
	pi.Trade = make([][]string, TradeSlotsV2)
	for slot := 0; slot < TradeSlotsV2; slot++ {
		row := make([]string, t)
		for k := 0; k < t; k++ {
			row[k] = s(int64(10+slot+k) * 1000000)
		}
		pi.Trade[slot] = row
	}
	for k := 0; k < t; k++ {
		pi.GridPrice = append(pi.GridPrice, s(100000))
		pi.FitPrice = append(pi.FitPrice, s(36000))
		pi.LossAlloc = append(pi.LossAlloc, s(1000))
	}
	for k := 0; k <= t; k++ {
		pi.SMin = append(pi.SMin, s(100000000))
		pi.SMax = append(pi.SMax, s(400000000))
	}
	return pi
}

// TestPublicInputV2FieldCount V2 字段计数 = 10 + TradeSlotsV2·T + 3T + 2(T+1),
// 且与 ToFields 一致
func TestPublicInputV2FieldCount(t *testing.T) {
	for _, tt := range []int{4, 24} {
		pi := samplePublicInputV2(tt)
		fields, err := PublicInputV2ToFields(pi)
		if err != nil {
			t.Fatalf("ToFields: %v", err)
		}
		want := PublicInputV2FieldCount(tt)
		if len(fields) != want {
			t.Fatalf("T=%d: 域元素数 %d ≠ 契约 %d", tt, len(fields), want)
		}
	}
}

// TestPublicInputV2Order 前三个字段必须是 session/version/cost (顺序冻结)
func TestPublicInputV2Order(t *testing.T) {
	pi := samplePublicInputV2(4)
	fields, err := PublicInputV2ToFields(pi)
	if err != nil {
		t.Fatalf("ToFields: %v", err)
	}
	if fields[0].Int64() != 1234500000 || fields[1].Int64() != 2 {
		t.Fatalf("V2 前两个字段顺序错乱: %v, %v", fields[0], fields[1])
	}
}

// TestValidatePublicInputV2 结构校验: 错版本号/错长度/负值应被拒
func TestValidatePublicInputV2(t *testing.T) {
	pi := samplePublicInputV2(4)
	if err := ValidatePublicInputV2(pi); err != nil {
		t.Fatalf("合法输入不应被拒: %v", err)
	}

	bad := samplePublicInputV2(4)
	bad.VersionTag = "1" // 跨版本重放
	if err := ValidatePublicInputV2(bad); err == nil {
		t.Fatal("错版本号应被拒")
	}

	bad2 := samplePublicInputV2(4)
	bad2.GridPrice = append(bad2.GridPrice, "100000") // 长度错
	if err := ValidatePublicInputV2(bad2); err == nil {
		t.Fatal("向量长度错误应被拒")
	}

	bad3 := samplePublicInputV2(4)
	bad3.Trade[0][0] = "-5000000" // 负公开输入
	if err := ValidatePublicInputV2(bad3); err == nil {
		t.Fatal("负公开输入应被拒")
	}

	bad4 := samplePublicInputV2(4)
	bad4.Trade = bad4.Trade[:TradeSlotsV2-1] // 槽位数不足
	if err := ValidatePublicInputV2(bad4); err == nil {
		t.Fatal("交易行槽位数错误应被拒")
	}
}

// TestCheckTradeRowAgainstMatrix 交叉核对: 槽位映射/非成员补零/未绑定交易拒绝
func TestCheckTradeRowAgainstMatrix(t *testing.T) {
	const nMG, slots = 3, 4
	// 载荷矩阵 [N][N][T] (反对称)
	p := make([][][]float64, nMG)
	for i := range p {
		p[i] = make([][]float64, nMG)
		for j := range p[i] {
			p[i][j] = make([]float64, slots)
		}
	}
	p[0][1][0], p[1][0][0] = 5, -5 // MG1↔MG2
	p[0][2][1], p[2][0][1] = -2, 2 // MG1↔MG3
	p[1][2][2], p[2][1][2] = 3, -3 // MG2↔MG3

	// MG1 的公开行: 槽0↔成员2, 槽1↔成员3, 其余槽位必须为 0
	row := make([][]string, TradeSlotsV2)
	for s := range row {
		row[s] = make([]string, slots)
		for k := 0; k < slots; k++ {
			row[s][k] = "0"
		}
	}
	for k := 0; k < slots; k++ {
		row[0][k] = FloatToField(p[0][1][k]).String() // 槽0 ↔ 成员2
		row[1][k] = FloatToField(p[0][2][k]).String() // 槽1 ↔ 成员3
	}
	if err := checkTradeRowAgainstMatrix(0, p[0], row, nMG, slots); err != nil {
		t.Fatalf("合法行不应被拒: %v", err)
	}

	// 环流攻击: 保持净进口不变, 但把 (MG1,MG2) 的交易挪到 (MG1,MG3) —
	// 载荷矩阵相应改动, 公开行未跟着改 → 必须被拒
	shifted := make([][][]float64, nMG)
	for i := range shifted {
		shifted[i] = make([][]float64, nMG)
		for j := range shifted[i] {
			shifted[i][j] = append([]float64(nil), p[i][j]...)
		}
	}
	delta := 1.0
	shifted[0][1][0] += delta
	shifted[1][0][0] -= delta
	shifted[0][2][0] -= delta
	shifted[2][0][0] += delta
	if err := checkTradeRowAgainstMatrix(0, shifted[0], row, nMG, slots); err == nil {
		t.Fatal("载荷矩阵被改动而公开行未随之改动, 应被拒")
	}

	// 非成员槽位非零 (结算规模 N=2 时槽1 必须为 0) → 必须被拒
	if err := checkTradeRowAgainstMatrix(0, p[0][:2], row, 2, slots); err == nil {
		t.Fatal("非成员槽位非零应被拒")
	}
}

// TestCheckLineCapacity 线路容量链码校验 (V2 新增)
func TestCheckLineCapacity(t *testing.T) {
	nMG, slots := 3, 4
	trades := make([]float64, nMG*nMG*slots)
	caps := []float64{500, 400, 400} // (0,1),(0,2),(1,2)
	trades[0*nMG*slots+1*slots+0] = 300
	if err := CheckLineCapacity(trades, caps, nMG, slots); err != nil {
		t.Fatalf("合法交易不应被拒: %v", err)
	}
	trades[0*nMG*slots+1*slots+2] = 600
	if err := CheckLineCapacity(trades, caps, nMG, slots); err == nil {
		t.Fatal("超容量交易应被拒")
	}
	trades[0*nMG*slots+1*slots+2] = -600
	if err := CheckLineCapacity(trades, caps, nMG, slots); err == nil {
		t.Fatal("负向超容量交易应被拒")
	}
}
