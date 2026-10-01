package main

// ============================================================
// 补实验：攻击场景实测（论文评审意见 #5）
//
// 论文安全表声称以下攻击被拒，本测试逐一实测:
//   攻击1: 篡改交易量 (pGlobal 增加 5 kW)
//   攻击2: 篡改价格   (p2pPrice 提高 0.1 元/kWh)
//   攻击3: 重放 (同 session 二次结算)
//   攻击4: 篡改证明 (cr 末字节翻转)
//
// 使用真实 gnark 证明 (pb 电路, N=3,T=4, 真实 ADMM witness)。
// 运行: go test -run TestAttackScenarios -v
// ============================================================

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// loadRealArtifacts 加载 zkp_circuit 的真实 VK/proof/settle
func loadRealArtifacts(t *testing.T) (VerificationKey, ZKPProof, map[string]interface{}) {
	zkpDir := "../../../zkp_circuit"
	vkData, err := os.ReadFile(zkpDir + "/vk.json")
	if err != nil {
		t.Skip("zkp_circuit vk.json 缺失, 跳过攻击测试 (先运行 TestBenchReal 或 gen_witness)")
	}
	pfData, _ := os.ReadFile(zkpDir + "/proof.json")
	spData, _ := os.ReadFile(zkpDir + "/settle_payload.json")
	var vk VerificationKey
	var proof ZKPProof
	var sp map[string]interface{}
	json.Unmarshal(vkData, &vk)
	json.Unmarshal(pfData, &proof)
	json.Unmarshal(spData, &sp)
	return vk, proof, sp
}

// nestedToFloat3D 将 JSON 解析的 []interface{} 嵌套转 [][][]float64
func nestedToFloat3D(v interface{}) ([][][]float64, error) {
	a, ok := v.([]interface{})
	if !ok {
		return nil, fmt.Errorf("not a nested array")
	}
	out := make([][][]float64, len(a))
	for i, x := range a {
		b, ok := x.([]interface{})
		if !ok {
			return nil, fmt.Errorf("level-2 not array")
		}
		out[i] = make([][]float64, len(b))
		for j, y := range b {
			c, ok := y.([]interface{})
			if !ok {
				return nil, fmt.Errorf("level-3 not array")
			}
			out[i][j] = make([]float64, len(c))
			for k, z := range c {
				f, ok := z.(float64)
				if !ok {
					return nil, fmt.Errorf("not float64: %T", z)
				}
				out[i][j][k] = f
			}
		}
	}
	return out, nil
}

func TestAttackScenarios(t *testing.T) {
	vk, proof, sp := loadRealArtifacts(t)

	// 部署 VK
	stub, ctx := newTestChaincode()
	setupTestLedger(stub, ctx)
	contract := &EnergyTradingContract{}
	if err := runTx(stub, func() error {
		return contract.SetADMMVerificationKey(ctx, mustJSON(vk))
	}); err != nil {
		t.Fatalf("SetADMMVerificationKey 失败: %v", err)
	}

	sessionID := sp["sessionID"].(string)
	mgIDs := mustJSON(sp["microgridIDs"])
	pgOrig, err := nestedToFloat3D(sp["pGlobal"])
	if err != nil {
		t.Fatalf("解析 pGlobal 失败: %v", err)
	}
	ppOrig, err := nestedToFloat3D(sp["p2pPrice"])
	if err != nil {
		t.Fatalf("解析 p2pPrice 失败: %v", err)
	}
	piOrig := sp["publicInput"]
	proofJSON := mustJSON(proof)

	// 深度拷贝 pGlobal/p2pPrice
	clone := func(src [][][]float64) [][][]float64 {
		dst := make([][][]float64, len(src))
		for i := range src {
			dst[i] = make([][]float64, len(src[i]))
			for j := range src[i] {
				dst[i][j] = append([]float64{}, src[i][j]...)
			}
		}
		return dst
	}

	t.Run("Attack1_TamperedVolume", func(t *testing.T) {
		pg := clone(pgOrig)
		// 在 (0,1,0) 上增加 5 kW
		pg[0][1][0] += 5.0
		err := runTx(stub, func() error {
			return contract.SettleADMM(ctx, sessionID+"-att1", mgIDs, mustJSON(pg), mustJSON(ppOrig), proofJSON, mustJSON(piOrig))
		})
		if err == nil {
			t.Fatal("攻击1 失败: 篡改交易量未被拒绝")
		}
		t.Logf("[OK] 攻击1 篡改交易量 (+5 kW) 被拒绝: %v", err)
	})

	t.Run("Attack2_TamperedPrice", func(t *testing.T) {
		pp := clone(ppOrig)
		// 提高 (0,1,0) 价格 0.1 元
		pp[0][1][0] += 0.1
		err := runTx(stub, func() error {
			return contract.SettleADMM(ctx, sessionID+"-att2", mgIDs, mustJSON(pgOrig), mustJSON(pp), proofJSON, mustJSON(piOrig))
		})
		if err == nil {
			t.Fatal("攻击2 失败: 篡改价格未被拒绝")
		}
		t.Logf("[OK] 攻击2 篡改价格 (+0.1 元) 被拒绝: %v", err)
	})

	t.Run("Attack3_Replay", func(t *testing.T) {
		// 给微网充值（真实交易量大于默认初始余额）
		stub.MockTransactionStart("recharge")
		for _, id := range []string{"MG1", "MG2", "MG3"} {
			mg, err := contract.GetMicroGrid(ctx, id)
			if err != nil {
				t.Fatalf("GetMicroGrid %s 失败: %v", id, err)
			}
			mg.Balance += 100_000_000
			mg.EnergyBalance += 100_000_000
			mgJSON, _ := json.Marshal(mg)
			key, _ := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{id})
			if err := ctx.GetStub().PutState(key, mgJSON); err != nil {
				t.Fatalf("充值 %s 失败: %v", id, err)
			}
		}
		stub.MockTransactionEnd("recharge")

		// 首次合法结算
		if err := runTx(stub, func() error {
			return contract.SettleADMM(ctx, sessionID, mgIDs, mustJSON(pgOrig), mustJSON(ppOrig), proofJSON, mustJSON(piOrig))
		}); err != nil {
			t.Fatalf("合法结算失败: %v", err)
		}
		// 同 session 重放
		err := runTx(stub, func() error {
			return contract.SettleADMM(ctx, sessionID, mgIDs, mustJSON(pgOrig), mustJSON(ppOrig), proofJSON, mustJSON(piOrig))
		})
		if err == nil {
			t.Fatal("攻击3 失败: 重放未被拒绝")
		}
		t.Logf("[OK] 攻击3 重放被拒绝: %v", err)
	})

	t.Run("Attack4_TamperedProof", func(t *testing.T) {
		// cr 末字节翻转
		p := proof
		p.Cr = append([]byte{}, proof.Cr...)
		p.Cr[len(p.Cr)-1] ^= 0x01
		err := runTx(stub, func() error {
			return contract.SettleADMM(ctx, sessionID+"-att4", mgIDs, mustJSON(pgOrig), mustJSON(ppOrig), mustJSON(p), mustJSON(piOrig))
		})
		if err == nil {
			t.Fatal("攻击4 失败: 篡改证明未被拒绝")
		}
		t.Logf("[OK] 攻击4 篡改证明被拒绝: %v", err)
	})
}
