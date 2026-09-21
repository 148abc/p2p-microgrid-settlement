package main

import (
	"encoding/json"
	"math/rand"
	"os"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

// ============================================================
// ============================================================

type mgWitnessInt struct {
	trade                                    [][]int64
	netTrade, gridPrice, fitPrice, lossAlloc []int64
	gridCap, dgCap, essPower                 int64
	etaNum, etaDen                           int64
	sMin, sMax                               []int64 // [T+1]
	dgCost, wearCost                         int64
	costClaim                                int64
	sessionHash, versionTag                  int64
	load, pv, wind, curtail        []int64
	pGrid, pExport, pDg, pCh, pDis []int64
	soc                            []int64 // [T+1]
	z                              []int64
}

const v2EtaNum, v2EtaDen = 19, 20

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func resolveBalanceAndCost(w *mgWitnessInt) {
	t := len(w.load)
	for k := 0; k < t; k++ {
		v := w.load[k] + w.lossAlloc[k] + w.pExport[k] -
			(w.pv[k] + w.wind[k]) - (w.pDis[k] - w.pCh[k]) -
			w.netTrade[k] - w.pDg[k]
		if v < 0 {
			d := -v
			take := min64(d, w.pv[k])
			w.pv[k] -= take
			d -= take
			take = min64(d, w.wind[k])
			w.wind[k] -= take
			d -= take
			if d > 0 {
				w.pExport[k] += d
			}
			v = 0
		}
		if v > w.gridCap {
			w.wind[k] += v - w.gridCap
			v = w.gridCap
		}
		w.pGrid[k] = v
	}
	cost := int64(0)
	for k := 0; k < t; k++ {
		cost += w.gridPrice[k] * (w.pGrid[k] + w.lossAlloc[k])
		cost += w.dgCost * w.pDg[k]
		cost += w.wearCost * w.pDis[k]
		cost -= w.fitPrice[k] * w.pExport[k]
	}
	w.costClaim = cost
}

func buildConsistentMG(rng *rand.Rand, t int) mgWitnessInt {
	const S = int64(1_000_000)
	w := mgWitnessInt{
		gridCap: 500 * S, dgCap: 200 * S, essPower: 300 * S,
		etaNum: v2EtaNum, etaDen: v2EtaDen,
		dgCost: 150_000, wearCost: 15_000,
		sessionHash: 1234500000, versionTag: CircuitVersionV2,
	}
	mk := func(n int) []int64 { return make([]int64, n) }
	w.netTrade, w.gridPrice, w.fitPrice, w.lossAlloc = mk(t), mk(t), mk(t), mk(t)
	w.trade = make([][]int64, TradeSlotsV2)
	for j := range w.trade {
		w.trade[j] = mk(t)
	}
	w.load, w.pv, w.wind, w.curtail = mk(t), mk(t), mk(t), mk(t)
	w.pGrid, w.pExport, w.pDg, w.pCh, w.pDis, w.z = mk(t), mk(t), mk(t), mk(t), mk(t), mk(t)
	w.soc, w.sMin, w.sMax = mk(t+1), mk(t+1), mk(t+1)

	for k := 0; k < t; k++ {
		switch {
		case k%6 >= 4:
			w.gridPrice[k] = 200_000
		case k%6 >= 2:
			w.gridPrice[k] = 100_000
		default:
			w.gridPrice[k] = 45_000
		}
		if k%6 < 4 {
			w.fitPrice[k] = 36_000
		}
		w.lossAlloc[k] = int64(rng.Intn(4)) * S / 1000
	}

	U := int64(2_000_000)
	w.soc[0] = 250 * S
	for k := 0; k < t; k++ {
		w.load[k] = int64(200+rng.Intn(300)) * S
		w.pv[k] = int64(rng.Intn(150)) * S
		w.wind[k] = int64(rng.Intn(200)) * S
		w.pExport[k] = int64(rng.Intn(30)) * S
		w.pDg[k] = int64(rng.Intn(80)) * S
		for j := 0; j < TradeSlotsV2; j++ {
			w.trade[j][k] = int64(rng.Intn(11)-5) * S
		}
		switch {
		case k < t/2:
			w.pCh[k], w.z[k] = 20*U, 1
		case k >= (t+1)/2:
			w.pDis[k], w.z[k] = 361*U/20, 0
		}
	}
	for k := 0; k < t; k++ {
		row := int64(0)
		for j := 0; j < TradeSlotsV2; j++ {
			row += w.trade[j][k]
		}
		w.netTrade[k] = row
	}
	for k := 0; k < t; k++ {
		switch {
		case k < t/2:
			w.soc[k+1] = w.soc[k] + 19*U
		case k >= (t+1)/2:
			w.soc[k+1] = w.soc[k] - 19*U
		default:
			w.soc[k+1] = w.soc[k]
		}
	}
	lo, hi := w.soc[0], w.soc[0]
	for k := 1; k <= t; k++ {
		lo, hi = min64(lo, w.soc[k]), max64(hi, w.soc[k])
	}
	for k := 0; k <= t; k++ {
		w.sMin[k], w.sMax[k] = lo-10*S, hi+10*S
	}
	w.sMin[0], w.sMax[0] = w.soc[0]-10*S, w.soc[0]+10*S // 2·SOC[0]=SMin+SMax

	resolveBalanceAndCost(&w)
	return w
}

func assignMGv2(w *mgWitnessInt, tmpl bool) *CircuitMGv2 {
	varOf := func(xs []int64) []frontend.Variable {
		out := make([]frontend.Variable, len(xs))
		for i, x := range xs {
			if tmpl {
				out[i] = nil
			} else {
				out[i] = x
			}
		}
		return out
	}
	c := &CircuitMGv2{}
	if tmpl {
		c.Statement.SessionHash, c.Statement.VersionTag, c.Statement.CostClaim = nil, nil, nil
	} else {
		c.Statement.SessionHash, c.Statement.VersionTag, c.Statement.CostClaim = w.sessionHash, w.versionTag, w.costClaim
	}
	c.Statement.Trade = make([][]frontend.Variable, len(w.trade))
	for j := range w.trade {
		c.Statement.Trade[j] = varOf(w.trade[j])
	}
	c.Statement.GridPrice = varOf(w.gridPrice)
	c.Statement.FitPrice = varOf(w.fitPrice)
	c.Statement.LossAlloc = varOf(w.lossAlloc)
	if tmpl {
		c.Statement.GridCap, c.Statement.DgCap, c.Statement.EssPower = nil, nil, nil
		c.Statement.EtaNum, c.Statement.EtaDen = nil, nil
	} else {
		c.Statement.GridCap, c.Statement.DgCap, c.Statement.EssPower = w.gridCap, w.dgCap, w.essPower
		c.Statement.EtaNum, c.Statement.EtaDen = w.etaNum, w.etaDen
	}
	c.Statement.SMin = varOf(w.sMin)
	c.Statement.SMax = varOf(w.sMax)
	if tmpl {
		c.Statement.DgCost, c.Statement.WearCost = nil, nil
	} else {
		c.Statement.DgCost, c.Statement.WearCost = w.dgCost, w.wearCost
	}
	c.Dispatch.Load = varOf(w.load)
	c.Dispatch.PV = varOf(w.pv)
	c.Dispatch.Wind = varOf(w.wind)
	c.Dispatch.Curtail = varOf(w.curtail)
	c.Dispatch.PGrid = varOf(w.pGrid)
	c.Dispatch.PExport = varOf(w.pExport)
	c.Dispatch.PDg = varOf(w.pDg)
	c.Dispatch.PCh = varOf(w.pCh)
	c.Dispatch.PDis = varOf(w.pDis)
	c.Dispatch.SOC = varOf(w.soc)
	c.Dispatch.Z = varOf(w.z)
	return c
}

func compileProveVerify(t *testing.T, tmpl frontend.Circuit, c frontend.Circuit) {
	t.Helper()
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, tmpl)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	witness, err := frontend.NewWitness(c, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	proof, err := groth16.Prove(ccs, pk, witness)
	if err != nil {
		t.Fatalf("prove (合法 witness 应可满足): %v", err)
	}
	pub, err := frontend.NewWitness(c, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		t.Fatalf("public witness: %v", err)
	}
	if err := groth16.Verify(proof, vk, pub); err != nil {
		t.Fatalf("verify (合法证明应通过): %v", err)
	}
}

func TestMGv2HappyPath(t *testing.T) {
	rng := rand.New(rand.NewSource(20260902))
	w := buildConsistentMG(rng, 6)
	compileProveVerify(t, assignMGv2(&w, true), assignMGv2(&w, false))
}

func TestMGv2TamperReject(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(w *mgWitnessInt)
	}{
		{"ExportOverCap", func(w *mgWitnessInt) { w.pExport[0] = w.gridCap + 1 }},
		{"DgOverCap", func(w *mgWitnessInt) { w.pDg[0] = w.dgCap + 1 }},
		{"DischargeOverEssPower", func(w *mgWitnessInt) { w.pDis[0] = w.essPower + 19 }},
		{"NegativeGrid", func(w *mgWitnessInt) { w.pGrid[0] = -1_000_000 }},
		{"CurtailOverRenewable", func(w *mgWitnessInt) {
			w.curtail[0] = w.pv[0] + w.wind[0] + 1_000_000
		}},
		{"SOCJump", func(w *mgWitnessInt) { w.soc[1] += 1_000_000 }},
		{"SOCOutOfBand", func(w *mgWitnessInt) { w.soc[2] = w.sMax[2] + 1_000_000 }},
		{"BoundaryViolation", func(w *mgWitnessInt) {
			w.soc[len(w.soc)-1] += 1_000_000
		}},
		{"MutexViolation", func(w *mgWitnessInt) { w.z[0] = 2 }},
		{"BadCostClaim", func(w *mgWitnessInt) { w.costClaim += 1_000_001 }},
		{"WrongVersionTag", func(w *mgWitnessInt) { w.versionTag = 1 }},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(20260902))
			w := buildConsistentMG(rng, 6)
			tc.mutate(&w)
			tmpl := assignMGv2(&w, true)
			c := assignMGv2(&w, false)
			ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, tmpl)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			pk, _, err := groth16.Setup(ccs)
			if err != nil {
				t.Fatalf("setup: %v", err)
			}
			witness, err := frontend.NewWitness(c, ecc.BN254.ScalarField())
			if err != nil {
				return
			}
			if _, err := groth16.Prove(ccs, pk, witness); err == nil {
				t.Fatalf("篡改 (%s) 应证明失败, 却成功了", tc.name)
			}
		})
	}
}

func buildFleetV2(blocks []mgWitnessInt, trades [][][]int64, tmpl bool) *CircuitFleetV2 {
	n := len(blocks)
	varOf := func(xs []int64) []frontend.Variable {
		out := make([]frontend.Variable, len(xs))
		for i, x := range xs {
			if tmpl {
				out[i] = nil
			} else {
				out[i] = x
			}
		}
		return out
	}
	scalar := func(x int64) frontend.Variable {
		if tmpl {
			return nil
		}
		return x
	}
	c := &CircuitFleetV2{
		SessionHash: scalar(9876500000),
		VersionTag:  scalar(CircuitVersionV2),
		PGlobal:     make([][][]frontend.Variable, n),
		MGs:         make([]MGFleetBlockV2, n),
	}
	for i := 0; i < n; i++ {
		c.PGlobal[i] = make([][]frontend.Variable, n)
		for j := 0; j < n; j++ {
			c.PGlobal[i][j] = varOf(trades[i][j])
		}
		w := &blocks[i]
		blk := &c.MGs[i]
		blk.CostClaim = scalar(w.costClaim)
		blk.GridPrice, blk.FitPrice, blk.LossAlloc = varOf(w.gridPrice), varOf(w.fitPrice), varOf(w.lossAlloc)
		blk.GridCap, blk.DgCap, blk.EssPower = scalar(w.gridCap), scalar(w.dgCap), scalar(w.essPower)
		blk.EtaNum, blk.EtaDen = scalar(w.etaNum), scalar(w.etaDen)
		blk.SMin, blk.SMax = varOf(w.sMin), varOf(w.sMax)
		blk.DgCost, blk.WearCost = scalar(w.dgCost), scalar(w.wearCost)
		blk.Dispatch = MGDispatchV2{
			Load: varOf(w.load), PV: varOf(w.pv), Wind: varOf(w.wind),
			Curtail: varOf(w.curtail), PGrid: varOf(w.pGrid),
			PExport: varOf(w.pExport), PDg: varOf(w.pDg),
			PCh: varOf(w.pCh), PDis: varOf(w.pDis),
			SOC: varOf(w.soc), Z: varOf(w.z),
		}
	}
	return c
}

func TestFleetV2HappyPath(t *testing.T) {
	rng := rand.New(rand.NewSource(20260903))
	tt, n := 4, 2
	const S = int64(1_000_000)

	trades := make([][][]int64, n)
	for i := range trades {
		trades[i] = make([][]int64, n)
		for j := range trades[i] {
			trades[i][j] = make([]int64, tt)
		}
	}
	for k := 0; k < tt; k++ {
		v := int64(rng.Intn(50)) * S
		trades[0][1][k], trades[1][0][k] = v, -v
	}

	blocks := make([]mgWitnessInt, n)
	for i := 0; i < n; i++ {
		blocks[i] = buildConsistentMG(rng, tt)
		for k := 0; k < tt; k++ {
			row := int64(0)
			for j := 0; j < n; j++ {
				row += trades[i][j][k]
			}
			blocks[i].netTrade[k] = row
		}
		blocks[i].curtail = make([]int64, tt)
		resolveBalanceAndCost(&blocks[i])
	}
	compileProveVerify(t, buildFleetV2(blocks, trades, true), buildFleetV2(blocks, trades, false))
}

func TestProveV2FromJSON(t *testing.T) {
	path := os.Getenv("WITNESS_V2")
	if path == "" {
		t.Skip("WITNESS_V2 未设置, 跳过 e2e")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read witness: %v", err)
	}
	var doc struct {
		Public struct {
			SessionHash int64     `json:"session_hash"`
			CostClaim   int64     `json:"cost_claim"`
			Trade       [][]int64 `json:"trade"`
			GridPrice   []int64   `json:"grid_price"`
			FitPrice    []int64 `json:"fit_price"`
			LossAlloc   []int64 `json:"loss_alloc"`
			GridCap     int64   `json:"grid_cap"`
			DgCap       int64   `json:"dg_cap"`
			EssPower    int64   `json:"ess_power"`
			EtaNum      int64   `json:"eta_num"`
			EtaDen      int64   `json:"eta_den"`
			SMin        []int64 `json:"s_min"`
			SMax        []int64 `json:"s_max"`
			DgCost      int64   `json:"dg_cost"`
			WearCost    int64   `json:"wear_cost"`
		} `json:"public"`
		Private struct {
			Load    []int64 `json:"load"`
			PV      []int64 `json:"pv"`
			Wind    []int64 `json:"wind"`
			Curtail []int64 `json:"curtail"`
			PGrid   []int64 `json:"p_grid"`
			PExport []int64 `json:"p_export"`
			PDg     []int64 `json:"p_dg"`
			PCh     []int64 `json:"p_ch"`
			PDis    []int64 `json:"p_dis"`
			SOC     []int64 `json:"soc"`
			Z       []int64 `json:"z"`
		} `json:"private"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parse witness: %v", err)
	}
	w := mgWitnessInt{
		sessionHash: doc.Public.SessionHash, versionTag: CircuitVersionV2,
		costClaim: doc.Public.CostClaim,
		trade:     doc.Public.Trade, gridPrice: doc.Public.GridPrice,
		fitPrice: doc.Public.FitPrice, lossAlloc: doc.Public.LossAlloc,
		gridCap: doc.Public.GridCap, dgCap: doc.Public.DgCap,
		essPower: doc.Public.EssPower, etaNum: doc.Public.EtaNum,
		etaDen: doc.Public.EtaDen, sMin: doc.Public.SMin, sMax: doc.Public.SMax,
		dgCost: doc.Public.DgCost, wearCost: doc.Public.WearCost,
		load: doc.Private.Load, pv: doc.Private.PV, wind: doc.Private.Wind,
		curtail: doc.Private.Curtail, pGrid: doc.Private.PGrid,
		pExport: doc.Private.PExport, pDg: doc.Private.PDg, pCh: doc.Private.PCh,
		pDis: doc.Private.PDis, soc: doc.Private.SOC, z: doc.Private.Z,
	}
	tmpl := assignMGv2(&w, true)
	c := assignMGv2(&w, false)
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, tmpl)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	witness, err := frontend.NewWitness(c, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("witness unsatisfiable (真实调度应满足 v2 方程): %v", err)
	}
	proof, err := groth16.Prove(ccs, pk, witness)
	if err != nil {
		t.Fatalf("prove: %v", err)
	}
	pub, _ := frontend.NewWitness(c, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err := groth16.Verify(proof, vk, pub); err != nil {
		t.Fatalf("verify: %v", err)
	}
	t.Logf("v2 e2e OK: constraints=%d", ccs.GetNbConstraints())
	if outPath := os.Getenv("PROOF_OUT"); outPath != "" {
		out, _ := json.MarshalIndent(map[string]interface{}{
			"constraints": ccs.GetNbConstraints(), "witness": path,
			"verified": true,
		}, "", " ")
		if err := os.WriteFile(outPath, out, 0o644); err != nil {
			t.Fatalf("write proof out: %v", err)
		}
	}
}
