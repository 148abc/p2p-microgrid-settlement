package main

// ============================================================
// 补实验：扩展电路（PB+SOC / PB+COST）链上 SettleADMM 验证
//
// 证明"扩展电路同样可链上验证"（论文评审建议 #4）:
//   1. 加载 zkp_circuit 导出的 pb_soc / pb_cost 真实 VK + proof + settle
//   2. SetADMMVerificationKey 部署 VK（管理员）
//   3. SettleADMM 完整路径: 基础校验 -> 防重放 -> 会话绑定 ->
//      Groth16 配对验证 -> 结算派生 -> 原子账本结算
//
// 产物来源: zkp_circuit 的 TestExportExtendedCircuits
// 运行: go test -run TestExtendedCircuitOnChain -v
// ============================================================

import (
	"encoding/json"
	"os"
	"testing"
)

func TestExtendedCircuitOnChain(t *testing.T) {
	zkpDir := "../../../zkp_circuit"

	for _, family := range []string{"pb_soc", "pb_cost"} {
		t.Run(family, func(t *testing.T) {
			vkData, err := os.ReadFile(zkpDir + "/vk_" + family + "_3.json")
			if err != nil {
				t.Skipf("%s 产物缺失, 跳过 (先运行 zkp_circuit TestExportExtendedCircuits)", family)
			}
			pfData, _ := os.ReadFile(zkpDir + "/proof_" + family + "_3.json")
			spData, _ := os.ReadFile(zkpDir + "/settle_" + family + "_3.json")

			var vk VerificationKey
			var proof ZKPProof
			var sp struct {
				SessionID    string      `json:"sessionID"`
				MicrogridIDs []string    `json:"microgridIDs"`
				PGlobal      [][][]float64 `json:"pGlobal"`
				P2PPrice     [][][]float64 `json:"p2pPrice"`
				PublicInput  PublicInput `json:"publicInput"`
			}
			json.Unmarshal(vkData, &vk)
			json.Unmarshal(pfData, &proof)
			json.Unmarshal(spData, &sp)

			// MockStub 链码环境
			stub, ctx := newTestChaincode()
			setupTestLedger(stub, ctx)
			contract := &EnergyTradingContract{}

			// 1. 管理员部署 VK
			if err := runTx(stub, func() error {
				return contract.SetADMMVerificationKey(ctx, mustJSON(vk))
			}); err != nil {
				t.Fatalf("[%s] SetADMMVerificationKey 失败: %v", family, err)
			}

			// 1.5 给微网充值（合成 witness 的交易量大于默认初始余额）
			stub.MockTransactionStart("recharge")
			for _, id := range sp.MicrogridIDs {
				mg, err := contract.GetMicroGrid(ctx, id)
				if err != nil {
					t.Fatalf("[%s] GetMicroGrid %s 失败: %v", family, id, err)
				}
				mg.Balance += 100_000_000 // +100万元
				mg.EnergyBalance += 100_000_000 // +10万kWh
				mgJSON, _ := json.Marshal(mg)
				key, _ := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{id})
				if err := ctx.GetStub().PutState(key, mgJSON); err != nil {
					t.Fatalf("[%s] 充值 %s 失败: %v", family, id, err)
				}
			}
			stub.MockTransactionEnd("recharge")

			// 2. SettleADMM 完整路径（真实 gnark 证明 + 扩展电路 VK）
			if err := runTx(stub, func() error {
				return contract.SettleADMM(ctx,
					sp.SessionID,
					mustJSON(sp.MicrogridIDs),
					mustJSON(sp.PGlobal),
					mustJSON(sp.P2PPrice),
					mustJSON(proof),
					mustJSON(sp.PublicInput))
			}); err != nil {
				t.Fatalf("[%s] SettleADMM 失败: %v", family, err)
			}

			// 3. 防重放确认（同一 session 第二次被拒）
			if err := runTx(stub, func() error {
				return contract.SettleADMM(ctx,
					sp.SessionID,
					mustJSON(sp.MicrogridIDs),
					mustJSON(sp.PGlobal),
					mustJSON(sp.P2PPrice),
					mustJSON(proof),
					mustJSON(sp.PublicInput))
			}); err == nil {
				t.Fatalf("[%s] 防重放失败: 同一 session 二次结算未被拒绝", family)
			}

			t.Logf("[OK] %s 扩展电路链上验证+结算通过 (真实 gnark 证明, MockStub SettleADMM 完整路径)", family)
		})
	}
}
