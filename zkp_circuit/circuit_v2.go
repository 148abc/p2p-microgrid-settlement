package main

import (
	"fmt"

	"github.com/consensys/gnark/frontend"
)

// ============================================================
//
//      EtaDen·SOC' = EtaDen·SOC + EtaNum²·ch − EtaDen²·dis)
// ============================================================

const CircuitVersionV2 = 2

const MarketSizeMaxV2 = 20

const TradeSlotsV2 = MarketSizeMaxV2 - 1

type MGStatementV2 struct {
	SessionHash frontend.Variable     `gnark:",public"`
	VersionTag  frontend.Variable     `gnark:",public"`
	CostClaim   frontend.Variable     `gnark:",public"`
	Trade       [][]frontend.Variable `gnark:",public"`
	GridPrice   []frontend.Variable `gnark:",public"` // [T] ToU (×1e6$/kWh)
	FitPrice    []frontend.Variable `gnark:",public"` // [T]
	LossAlloc   []frontend.Variable `gnark:",public"` // [T]
	GridCap     frontend.Variable   `gnark:",public"`
	DgCap       frontend.Variable   `gnark:",public"`
	EssPower    frontend.Variable   `gnark:",public"`
	EtaNum      frontend.Variable   `gnark:",public"` // η = EtaNum/EtaDen
	EtaDen      frontend.Variable   `gnark:",public"`
	SMin        []frontend.Variable `gnark:",public"` // [T+1]
	SMax        []frontend.Variable `gnark:",public"` // [T+1]
	DgCost      frontend.Variable   `gnark:",public"` // $/kWh ×1e6
	WearCost    frontend.Variable   `gnark:",public"` // $/kWh ×1e6
}

type MGDispatchV2 struct {
	Load    []frontend.Variable // [T]
	PV      []frontend.Variable // [T]
	Wind    []frontend.Variable // [T]
	Curtail []frontend.Variable // [T]
	PGrid   []frontend.Variable // [T]
	PExport []frontend.Variable // [T]
	PDg     []frontend.Variable // [T]
	PCh     []frontend.Variable // [T]
	PDis    []frontend.Variable // [T]
	SOC     []frontend.Variable // [T+1]
	Z       []frontend.Variable
}

type CircuitMGv2 struct {
	Statement MGStatementV2
	Dispatch  MGDispatchV2
}

type CircuitFleetV2 struct {
	SessionHash frontend.Variable       `gnark:",public"`
	VersionTag  frontend.Variable       `gnark:",public"`
	PGlobal     [][][]frontend.Variable `gnark:",public"` // [N][N][T]
	MGs         []MGFleetBlockV2
}

type MGFleetBlockV2 struct {
	CostClaim frontend.Variable   `gnark:",public"`
	GridPrice []frontend.Variable `gnark:",public"` // [T]
	FitPrice  []frontend.Variable `gnark:",public"` // [T]
	LossAlloc []frontend.Variable `gnark:",public"` // [T]
	GridCap   frontend.Variable   `gnark:",public"`
	DgCap     frontend.Variable   `gnark:",public"`
	EssPower  frontend.Variable   `gnark:",public"`
	EtaNum    frontend.Variable   `gnark:",public"`
	EtaDen    frontend.Variable   `gnark:",public"`
	SMin      []frontend.Variable `gnark:",public"` // [T+1]
	SMax      []frontend.Variable `gnark:",public"` // [T+1]
	DgCost    frontend.Variable   `gnark:",public"`
	WearCost  frontend.Variable   `gnark:",public"`
	Dispatch  MGDispatchV2
}

func le64(api frontend.API, x, y frontend.Variable) {
	api.ToBinary(api.Sub(y, x), 64)
}

func applyMGv2(api frontend.API, s *MGStatementV2, netTrade []frontend.Variable,
	d *MGDispatchV2) error {
	t := len(netTrade)
	if len(d.SOC) != t+1 || len(s.SMin) != t+1 || len(s.SMax) != t+1 {
		return fmt.Errorf("v2: SOC/带长度应为 T+1=%d", t+1)
	}
	for _, arr := range [][]frontend.Variable{s.GridPrice, s.FitPrice,
		s.LossAlloc, d.Load, d.PV, d.Wind, d.Curtail, d.PGrid, d.PExport,
		d.PDg, d.PCh, d.PDis, d.Z} {
		if len(arr) != t {
			return fmt.Errorf("v2: 向量长度应为 T=%d", t)
		}
	}

	// load + curtail + loss + export = pv + wind + (dis − ch) + grid + netTrade + dg
	for k := 0; k < t; k++ {
		lhs := api.Add(
			api.Add(d.Load[k], d.Curtail[k]),
			api.Add(s.LossAlloc[k], d.PExport[k]),
		)
		gen := api.Add(d.PV[k], d.Wind[k])
		disCh := api.Sub(d.PDis[k], d.PCh[k])
		rhs := api.Add(
			api.Add(gen, disCh),
			api.Add(d.PGrid[k], api.Add(netTrade[k], d.PDg[k])),
		)
		api.AssertIsEqual(lhs, rhs)
	}

	ab := api.Mul(s.EtaNum, s.EtaDen)
	a2 := api.Mul(s.EtaNum, s.EtaNum)
	b2 := api.Mul(s.EtaDen, s.EtaDen)
	for k := 0; k < t; k++ {
		lhs := api.Mul(ab, d.SOC[k+1])
		rhs := api.Add(
			api.Mul(ab, d.SOC[k]),
			api.Sub(api.Mul(a2, d.PCh[k]), api.Mul(b2, d.PDis[k])),
		)
		api.AssertIsEqual(lhs, rhs)
	}

	api.AssertIsEqual(d.SOC[0], d.SOC[t])
	api.AssertIsEqual(api.Mul(2, d.SOC[0]), api.Add(s.SMin[0], s.SMax[0]))

	for k := 0; k < t; k++ {
		api.AssertIsEqual(api.Mul(d.Z[k], api.Sub(d.Z[k], 1)), 0)
		api.AssertIsEqual(api.Mul(d.PCh[k], api.Sub(1, d.Z[k])), 0)
		api.AssertIsEqual(api.Mul(d.PDis[k], d.Z[k]), 0)
	}

	for k := 0; k < t; k++ {
		le64(api, 0, d.PGrid[k])
		le64(api, d.PGrid[k], s.GridCap)
		le64(api, 0, d.PExport[k])
		le64(api, d.PExport[k], s.GridCap)
		le64(api, 0, d.PDg[k])
		le64(api, d.PDg[k], s.DgCap)
		le64(api, 0, d.PCh[k])
		le64(api, d.PCh[k], s.EssPower)
		le64(api, 0, d.PDis[k])
		le64(api, d.PDis[k], s.EssPower)
		le64(api, 0, d.Curtail[k])
		le64(api, d.Curtail[k], api.Add(d.PV[k], d.Wind[k]))
	}
	for k := 0; k <= t; k++ {
		le64(api, s.SMin[k], d.SOC[k])
		le64(api, d.SOC[k], s.SMax[k])
	}

	// CostClaim = Σ_k [ ToU·(grid+loss) + dgCost·dg + wearCost·dis − fit·export ]
	cost := frontend.Variable(0)
	for k := 0; k < t; k++ {
		cost = api.Add(cost, api.Mul(s.GridPrice[k],
			api.Add(d.PGrid[k], s.LossAlloc[k])))
		cost = api.Add(cost, api.Mul(s.DgCost, d.PDg[k]))
		cost = api.Add(cost, api.Mul(s.WearCost, d.PDis[k]))
		cost = api.Sub(cost, api.Mul(s.FitPrice[k], d.PExport[k]))
	}
	api.AssertIsEqual(s.CostClaim, cost)

	return nil
}

func (c *CircuitMGv2) Define(api frontend.API) error {
	if len(c.Dispatch.SOC) == 0 {
		return nil
	}
	api.AssertIsEqual(c.Statement.VersionTag, CircuitVersionV2)
	if len(c.Statement.Trade) != TradeSlotsV2 {
		return fmt.Errorf("v2: 交易行槽位数应为 %d, 实际 %d",
			TradeSlotsV2, len(c.Statement.Trade))
	}
	t := len(c.Dispatch.Load)
	netTrade := make([]frontend.Variable, t)
	for k := 0; k < t; k++ {
		s := frontend.Variable(0)
		for j := 0; j < TradeSlotsV2; j++ {
			if len(c.Statement.Trade[j]) != t {
				return fmt.Errorf("v2: 交易行列长度应为 T=%d", t)
			}
			s = api.Add(s, c.Statement.Trade[j][k])
		}
		netTrade[k] = s
	}
	return applyMGv2(api, &c.Statement, netTrade, &c.Dispatch)
}

func (c *CircuitFleetV2) Define(api frontend.API) error {
	n := len(c.MGs)
	if n == 0 || len(c.PGlobal) == 0 {
		return nil
	}
	api.AssertIsEqual(c.VersionTag, CircuitVersionV2)
	t := len(c.PGlobal[0][0])
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			for k := 0; k < t; k++ {
				api.AssertIsEqual(
					api.Add(c.PGlobal[i][j][k], c.PGlobal[j][i][k]), 0)
			}
		}
	}
	for i := 0; i < n; i++ {
		netTrade := make([]frontend.Variable, t)
		for k := 0; k < t; k++ {
			s := frontend.Variable(0)
			for j := 0; j < n; j++ {
				s = api.Add(s, c.PGlobal[i][j][k])
			}
			netTrade[k] = s
		}
		blk := &c.MGs[i]
		s := MGStatementV2{
			SessionHash: c.SessionHash,
			VersionTag:  c.VersionTag,
			CostClaim:   blk.CostClaim,
			GridPrice:   blk.GridPrice,
			FitPrice:    blk.FitPrice,
			LossAlloc:   blk.LossAlloc,
			GridCap:     blk.GridCap,
			DgCap:       blk.DgCap,
			EssPower:    blk.EssPower,
			EtaNum:      blk.EtaNum,
			EtaDen:      blk.EtaDen,
			SMin:        blk.SMin,
			SMax:        blk.SMax,
			DgCost:      blk.DgCost,
			WearCost:    blk.WearCost,
		}
		if err := applyMGv2(api, &s, netTrade, &blk.Dispatch); err != nil {
			return err
		}
	}
	return nil
}
