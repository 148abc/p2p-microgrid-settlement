package main

import (
	"encoding/json"
	"os"
	"testing"
	"time"
)

// TestBatchInvalidVerify 链码验证器对 100 个非法证明的批量处理:
// 全部拒绝 + 平均验证耗时 (链码真实 Verify 代码路径)
func TestBatchInvalidVerify(t *testing.T) {
	// 加载真实 VK + 真实证明 + 真实公开输入 (与 zkp_circuit 产物一致; 文件缺失则跳过)
	zkpDir := "../../../zkp_circuit"
	vkData, err := os.ReadFile(zkpDir + "/vk.json")
	if err != nil {
		t.Skip("zkp_circuit 产物缺失, 跳过批量验证测试")
	}
	pfData, _ := os.ReadFile(zkpDir + "/proof.json")
	spData, _ := os.ReadFile(zkpDir + "/settle_payload.json")
	var vk VerificationKey
	var proof ZKPProof
	var sp struct {
		PublicInput PublicInput `json:"publicInput"`
	}
	json.Unmarshal(vkData, &vk)
	json.Unmarshal(pfData, &proof)
	json.Unmarshal(spData, &sp)

	// 100 个非法证明: cr 末字节翻转 (与链上批量测试一致)
	var total time.Duration
	rejected := 0
	for i := 0; i < 100; i++ {
		p := proof
		p.Cr = append([]byte{}, proof.Cr...)
		p.Cr[len(p.Cr)-1] ^= 0x01
		start := time.Now()
		ok, err := VerifyADMMProof(p, sp.PublicInput, vk)
		total += time.Since(start)
		if err == nil && ok {
			t.Errorf("非法证明 #%d 意外通过", i)
		} else if !ok || err != nil {
			rejected++
		}
	}
	avgMs := float64(total.Microseconds()) / 100.0 / 1000.0
	t.Logf("100 个非法证明: 拒绝 %d/100, 平均链码验证耗时 %.3f ms (含反序列化失败路径)", rejected, avgMs)
	if rejected != 100 {
		t.Fatalf("应有 100 个被拒绝, 实际 %d", rejected)
	}
}
