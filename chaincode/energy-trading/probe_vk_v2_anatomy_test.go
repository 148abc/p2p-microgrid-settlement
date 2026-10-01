package main

// ============================================================
// 探针: 合约侧 V2 验证成本解剖 (分段计时)
//
// 背景: 2026-09-18 逐对交易行并入语句后, 公开输入 156 → 588 域元素,
// IC 从 157 → 589 个曲线点, 合约侧 verify 从 ~14 ms/证 涨到 ~30 ms/证。
// 本探针把一次验证拆成三段计时, 定位成本到底在哪:
//   (a) 证明点反序列化
//   (b) IC commitment: 逐点 ScalarMultiplication (旧实现) vs MultiExp (新实现)
//   (c) 四对配对检查
// 只测时序, 不影响生产路径。
//
// 运行: go test -run TestProbeVKV2Anatomy -v .
// ============================================================

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

func TestProbeVKV2Anatomy(t *testing.T) {
	vk, req := loadV2FixturesFrom(t, filepath.Join("testdata", "n10"))

	var rawVK VerificationKey
	b, err := os.ReadFile(filepath.Join("testdata", "n10", "vk_v2_mg.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &rawVK); err != nil {
		t.Fatal(err)
	}

	t0 := time.Now()
	verifier, err := NewGroth16Verifier(vk)
	if err != nil {
		t.Fatal(err)
	}
	buildOnce := time.Since(t0)

	proof := req.Proofs[0]
	publics, err := ValidateAndParsePublicInputV2(req.PublicInputs[0])
	if err != nil {
		t.Fatal(err)
	}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000.0 }

	// (a) 证明点反序列化 / (b1) 旧 IC / (b2) 新 IC / (c) 配对: 各测 5 次取中位
	// (单发计时在 Windows 上被 GC 与调度尖峰污染, 曾观察到 3~4 倍离群)
	med := func(xs []float64) float64 {
		for i := 0; i < len(xs); i++ {
			for j := i + 1; j < len(xs); j++ {
				if xs[j] < xs[i] {
					xs[i], xs[j] = xs[j], xs[i]
				}
			}
		}
		return xs[len(xs)/2]
	}
	var desers, naives, batches, pairings []float64
	for r := 0; r < 5; r++ {
		t0 = time.Now()
		var A, C bn254.G1Affine
		var B bn254.G2Affine
		if _, err := A.SetBytes(proof.Ar); err != nil {
			t.Fatal(err)
		}
		if _, err := B.SetBytes(proof.Bs); err != nil {
			t.Fatal(err)
		}
		if _, err := C.SetBytes(proof.Cr); err != nil {
			t.Fatal(err)
		}
		desers = append(desers, ms(time.Since(t0)))

		// (b1) IC commitment: 旧实现 (逐点标量乘)
		t0 = time.Now()
		var icNaive bn254.G1Affine
		icNaive.Set(&verifier.icG1[0])
		for i, x := range publics {
			var term bn254.G1Affine
			term.ScalarMultiplication(&verifier.icG1[i+1], x)
			var sum bn254.G1Affine
			sum.Add(&icNaive, &term)
			icNaive.Set(&sum)
		}
		naives = append(naives, ms(time.Since(t0)))

		// (b2) IC commitment: 新实现 (批量 MSM)
		t0 = time.Now()
		scalars := make([]fr.Element, len(publics))
		for i, x := range publics {
			scalars[i].SetBigInt(x)
		}
		var icBatch bn254.G1Affine
		if _, err := icBatch.MultiExp(verifier.icG1[1:], scalars, ecc.MultiExpConfig{}); err != nil {
			t.Fatal(err)
		}
		icBatch.Add(&icBatch, &verifier.icG1[0])
		batches = append(batches, ms(time.Since(t0)))
		if !icBatch.Equal(&icNaive) {
			t.Fatal("两种 IC commitment 结果不一致")
		}

		// (c) 配对检查 (新旧实现共用)
		t0 = time.Now()
		var na, nic, nc bn254.G1Affine
		na.Neg(&verifier.alphaG1)
		nic.Neg(&icBatch)
		nc.Neg(&C)
		_, err = bn254.PairingCheck([]bn254.G1Affine{A, na, nic, nc},
			[]bn254.G2Affine{B, verifier.betaG2, verifier.gammaG2, verifier.deltaG2})
		if err != nil {
			t.Fatal(err)
		}
		pairings = append(pairings, ms(time.Since(t0)))
	}
	deser, naive, batch, pairing := med(desers), med(naives), med(batches), med(pairings)

	fmt.Printf("\n[anatomy] 公开输入 %d 域元素, IC %d 点, VK 构建一次 %.1f ms\n",
		len(publics), len(verifier.icG1), buildOnce.Seconds()*1000)
	fmt.Printf("[anatomy]   (a) 证明反序列化        %6.2f ms\n", deser)
	fmt.Printf("[anatomy]   (b1) IC 逐点标量乘(旧)  %6.2f ms\n", naive)
	fmt.Printf("[anatomy]   (b2) IC 批量 MSM (新)   %6.2f ms\n", batch)
	fmt.Printf("[anatomy]   (c) 四对配对检查        %6.2f ms\n", pairing)
	fmt.Printf("[anatomy]   旧路径合计 ≈ %.2f ms/证 (含每证重建 VK)\n", deser+naive+pairing+buildOnce.Seconds()*1000)
	fmt.Printf("[anatomy]   新路径合计 ≈ %.2f ms/证 (VK 一次构建摊薄后)\n", deser+batch+pairing)
}
