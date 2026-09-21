package main

import (
	"encoding/json"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

// ============================================================
// ============================================================

func medianMs(xs []float64) float64 {
	for i := 0; i < len(xs); i++ {
		for j := i + 1; j < len(xs); j++ {
			if xs[j] < xs[i] {
				xs[i], xs[j] = xs[j], xs[i]
			}
		}
	}
	return xs[len(xs)/2]
}


func benchOne(t *testing.T, tmpl, assignment frontend.Circuit, runs int) map[string]interface{} {
	t.Helper()
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, tmpl)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	pub, _ := frontend.NewWitness(assignment, ecc.BN254.ScalarField(), frontend.PublicOnly())

	var proveMs, verifyMs []float64
	var proof groth16.Proof
	for r := 0; r < runs; r++ {
		t0 := time.Now()
		proof, err = groth16.Prove(ccs, pk, witness)
		if err != nil {
			t.Fatalf("prove run %d: %v", r, err)
		}
		proveMs = append(proveMs, float64(time.Since(t0).Microseconds())/1000.0)
		t0 = time.Now()
		if err := groth16.Verify(proof, vk, pub); err != nil {
			t.Fatalf("verify run %d: %v", r, err)
		}
		verifyMs = append(verifyMs, float64(time.Since(t0).Microseconds())/1000.0)
	}
	var pbn interface{ MarshalSolidity() []byte }
	sizeB := 0
	if p, ok := proof.(interface{ MarshalSolidity() []byte }); ok {
		pbn = p
		sizeB = len(p.MarshalSolidity())
	}
	_ = pbn
	sort.Float64s(proveMs)
	sort.Float64s(verifyMs)
	q := func(xs []float64, f float64) float64 {
		i := int(f * float64(len(xs)-1) + 0.5)
		return xs[i]
	}
	return map[string]interface{}{
		"constraints":       ccs.GetNbConstraints(),
		"prove_ms_median":   medianMs(proveMs),
		"verify_ms_median":  medianMs(verifyMs),
		"prove_ms_min":      proveMs[0],
		"prove_ms_max":      proveMs[len(proveMs)-1],
		"prove_ms_iqr":      [2]float64{q(proveMs, 0.25), q(proveMs, 0.75)},
		"verify_ms_min":     verifyMs[0],
		"verify_ms_max":     verifyMs[len(verifyMs)-1],
		"verify_ms_iqr":     [2]float64{q(verifyMs, 0.25), q(verifyMs, 0.75)},
		"proof_bytes":       sizeB,
		"runs":              runs,
		"verified":          true,
	}
}

func TestBenchV2(t *testing.T) {
	runs := 5
	if v := os.Getenv("BENCH_V2_RUNS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			runs = n
		}
	}
	out := map[string]interface{}{"config": map[string]interface{}{
		"T": 24, "runs": runs, "curve": "BN254/Groth16", "family": "v2",
	}, "results": []map[string]interface{}{}}

	// ---- per-MG (n=1), T=24 ----
	rng := rand.New(rand.NewSource(20260904))
	w := buildConsistentMG(rng, 24)
	res := benchOne(t, assignMGv2(&w, true), assignMGv2(&w, false), runs)
	res["circuit"] = "v2_mg"
	res["N"] = 1
	out["results"] = append(out["results"].([]map[string]interface{}), res)
	t.Logf("v2_mg T=24: %v", res)

	// ---- fleet N ∈ {3,10,20}, T=24 ----
	sizes := []int{3, 10, 20}
	if os.Getenv("BENCH_V2_ONLY_MG") != "" {
		sizes = nil
	}
	for _, n := range sizes {
		trades := make([][][]int64, n)
		for i := range trades {
			trades[i] = make([][]int64, n)
			for j := range trades[i] {
				trades[i][j] = make([]int64, 24)
			}
		}
		blocks := make([]mgWitnessInt, n)
		for i := 0; i < n; i++ {
			blocks[i] = buildConsistentMG(rand.New(rand.NewSource(int64(20260904+i))), 24)
		}
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				for k := 0; k < 24; k++ {
					lo, hi := int64(-80)*1_000_000, int64(80)*1_000_000
					v := lo + rand.New(rand.NewSource(
						int64(1000*i+j)*int64(97+k))).Int63()%(hi-lo)
					trades[i][j][k] = v
					trades[j][i][k] = -v
				}
			}
		}
		for i := 0; i < n; i++ {
			for k := 0; k < 24; k++ {
				row := int64(0)
				for j := 0; j < n; j++ {
					row += trades[i][j][k]
				}
				blocks[i].netTrade[k] = row
			}
			blocks[i].curtail = make([]int64, 24)
			resolveBalanceAndCost(&blocks[i])
		}
		res := benchOne(t, buildFleetV2(blocks, trades, true),
			buildFleetV2(blocks, trades, false), 5)
		res["circuit"] = "v2_fleet"
		res["N"] = n
		out["results"] = append(out["results"].([]map[string]interface{}), res)
		t.Logf("v2_fleet N=%d: %v", n, res)
	}

	b, _ := json.MarshalIndent(out, "", " ")
	if p := os.Getenv("BENCH_V2_OUT"); p != "" {
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
		t.Logf("saved -> %s", p)
	}
}
