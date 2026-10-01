package main

// ============================================================
// 探针: 合约侧 V2 验证成本拆分 (每证重建验证器 vs 缓存复用)
//
// 动机: VerifyADMMProofV2 每次都调用 NewGroth16Verifier(vk), 而 VK 的 IC
// 点数 = 公开输入数 + 1。2026-09-18 逐对交易行并入语句后, 公开输入从 156
// 涨到 588 域元素, IC 从 157 涨到 589 个曲线点, 于是"每证重建"的开销随之
// 线性增长 —— 合约侧 verify 从 ~14 ms/证 涨到 ~30 ms/证, 而原生验证器
// (不含重建) 只需约 1 ms。本探针量化这部分浪费, 作为"是否缓存验证器"的
// 决策依据。只测时序, 不改动生产路径。
//
// 运行: go test -run TestProbeVKV2CostSplit -v .
// ============================================================

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"
)

func TestProbeVKV2CostSplit(t *testing.T) {
	vk, req := loadV2FixturesFrom(t, filepath.Join("testdata", "n10"))
	n := len(req.Proofs)

	// A: 现行路径 —— 每份证明重建一次验证器
	t0 := time.Now()
	for i := 0; i < n; i++ {
		ok, err := VerifyADMMProofV2(req.Proofs[i], req.PublicInputs[i], vk)
		if err != nil || !ok {
			t.Fatalf("现行路径 verify %d: ok=%v err=%v", i+1, ok, err)
		}
	}
	aMS := float64(time.Since(t0).Microseconds()) / 1000.0

	// B: 缓存路径 —— 建一次验证器, 全部证明复用
	t0 = time.Now()
	verifier, err := NewGroth16Verifier(vk)
	if err != nil {
		t.Fatal(err)
	}
	buildMS := float64(time.Since(t0).Microseconds()) / 1000.0
	t0 = time.Now()
	for i := 0; i < n; i++ {
		publics, err := ValidateAndParsePublicInputV2(req.PublicInputs[i])
		if err != nil {
			t.Fatal(err)
		}
		ok, err := verifier.Verify(req.Proofs[i], publics)
		if err != nil || !ok {
			t.Fatalf("缓存路径 verify %d: ok=%v err=%v", i+1, ok, err)
		}
	}
	bMS := float64(time.Since(t0).Microseconds()) / 1000.0

	fmt.Printf("\n[probe] VK 重建 vs 缓存 (N=%d, IC=%d 点):\n", n, n)
	fmt.Printf("[probe]   A 每证重建: %7.1f ms 合计 (%5.1f ms/证)\n", aMS, aMS/float64(n))
	fmt.Printf("[probe]   B 建一次  : %7.1f ms (%5.1f ms) + 复用 %6.2f ms 合计 (%5.2f ms/证)\n",
		buildMS+bMS, buildMS, bMS, bMS/float64(n))
	fmt.Printf("[probe]   每证浪费 : %7.1f ms (%.1f×)\n",
		(aMS-bMS-buildMS)/float64(n), aMS/(bMS+buildMS))
}
