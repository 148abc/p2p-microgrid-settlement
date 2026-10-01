package main

// ============================================================
// A3 补实验: 明文约束检查 vs Groth16 配对验证
//
// 论文论证: "选择 ZKP 的真正理由是隐私，而非性能——明文约束检查
// 与 Groth16 配对验证的耗时差距仅在毫秒级"
//
// 本测试量化链码内"直接遍历约束等式"（明文路径）的耗时，
// 与 Groth16 配对检查（5.02ms, chart_data_real.md §十）对比。
//
// 明文路径模拟: 对 N=3, T=4 的 24 个约束（12 对称 + 12 平衡）
// 逐等式做整数域比较，等价于链码若直接验证 witness 所需的最少工作。
// ============================================================

import (
	"fmt"
	"testing"
	"time"
)

// plaintextCheckConstraints 明文验证: 直接检查交易对称与功率平衡约束
// （witness 全公开时的链码路径，与 Groth16 配对检查做对比）
func plaintextCheckConstraints(p [3][3][4]int64, load, gen, dis, ch, grid [3][4]int64, loss [3][4]int64) bool {
	// 约束 1: 交易对称 P[i][j][t] + P[j][i][t] = 0 (12 约束)
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			for t := 0; t < 4; t++ {
				if p[i][j][t]+p[j][i][t] != 0 {
					return false
				}
			}
		}
	}
	// 约束 2: 功率平衡 (12 约束)
	for i := 0; i < 3; i++ {
		for t := 0; t < 4; t++ {
			netTrade := p[i][0][t] + p[i][1][t] + p[i][2][t]
			lhs := load[i][t] + loss[i][t]
			rhs := gen[i][t] + (dis[i][t] - ch[i][t]) + grid[i][t] + netTrade
			if lhs != rhs {
				return false
			}
		}
	}
	return true
}

// TestA3_PlaintextCheckBenchmark 明文约束检查基准
func TestA3_PlaintextCheckBenchmark(t *testing.T) {
	// 构造一个满足约束的整数域数据（模拟 witness 公开后的检查）
	var p [3][3][4]int64
	var load, gen, dis, ch, grid, loss [3][4]int64
	for i := 0; i < 3; i++ {
		for t := 0; t < 4; t++ {
			load[i][t] = 100 + int64(i*10)
			gen[i][t] = 60 + int64(t)
			grid[i][t] = 40
			loss[i][t] = 5
		}
	}
	// 对称: p[0][1] = -p[1][0] 等
	for t := 0; t < 4; t++ {
		p[0][1][t] = 15
		p[1][0][t] = -15
		p[0][2][t] = 10
		p[2][0][t] = -10
		p[1][2][t] = 5
		p[2][1][t] = -5
	}

	// 正确性: 约束必须满足
	if !plaintextCheckConstraints(p, load, gen, dis, ch, grid, loss) {
		// 调整 grid 使平衡满足（类似 P_grid 重调整）
		for i := 0; i < 3; i++ {
			for t := 0; t < 4; t++ {
				netTrade := p[i][0][t] + p[i][1][t] + p[i][2][t]
				grid[i][t] = load[i][t] + loss[i][t] - gen[i][t] - (dis[i][t] - ch[i][t]) - netTrade
			}
		}
		if !plaintextCheckConstraints(p, load, gen, dis, ch, grid, loss) {
			t.Fatal("fixture not satisfiable")
		}
	}

	// 基准: 10000 次明文检查取平均
	const rounds = 10000
	start := time.Now()
	for r := 0; r < rounds; r++ {
		if !plaintextCheckConstraints(p, load, gen, dis, ch, grid, loss) {
			t.Fatal("unexpected false")
		}
	}
	elapsed := time.Since(start)
	avgUs := float64(elapsed.Microseconds()) / float64(rounds)

	// Groth16 配对验证参考: 5.02ms = 5020us (chart_data_real.md §十)
	const groth16Us = 5020.0
	ratio := groth16Us / avgUs

	fmt.Printf("\n=== A3: Plaintext constraint check vs Groth16 pairing ===\n")
	fmt.Printf("Plaintext 24-constraint check: %.3f us/check (%d rounds)\n", avgUs, rounds)
	fmt.Printf("Groth16 pairing verification : %.0f us (measured, chart_data_real.md)\n", groth16Us)
	fmt.Printf("Ratio (Groth16 / plaintext)  : %.1fx\n", ratio)
	fmt.Printf("Conclusion: ZKP verification adds %.2f ms over plaintext per settlement\n",
		(groth16Us-avgUs)/1000.0)

	// 断言: 明文检查 < 100us（远低于配对验证 5020us）
	if avgUs > 100 {
		t.Errorf("plaintext check too slow: %.3f us", avgUs)
	}
}
