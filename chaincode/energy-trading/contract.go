package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// EnergyTradingContract 能源交易智能合约
type EnergyTradingContract struct {
	contractapi.Contract
}

// 信用评分系统常量
const (
	InitialCreditScore   = 100 // 初始信用分
	MinCreditScore       = 0   // 最低信用分
	MaxCreditScore       = 1000 // 最高信用分（允许成长空间）
	AccessThreshold      = 60  // 准入阈值（低于此分数禁止交易）
	WarningThreshold     = 70  // 警告阈值（低于此分数发出警告）
	SuccessReward        = 1   // 交易成功奖励分数
	FailurePenalty       = 10  // 交易失败惩罚分数
	ViolationPenalty     = 15  // 严重违约惩罚分数
)

// MicroGrid 微网账户信息
type MicroGrid struct {
	ID               string `json:"id"`               // 微网ID
	Name             string `json:"name"`             // 微网名称
	Balance          int64  `json:"balance"`          // 账户余额（分，1元=100分）
	LockedBalance    int64  `json:"lockedBalance"`    // 锁定余额（分）
	EnergyBalance    int64  `json:"energyBalance"`    // 能源余额（Wh，1kWh=1000Wh）
	LockedEnergy     int64  `json:"lockedEnergy"`     // 锁定能源（Wh）
	Organization     string `json:"organization"`     // 所属组织
	CreditScore      int    `json:"creditScore"`      // 信用评分（0-100）
	TotalTrades      int64  `json:"totalTrades"`      // 总交易次数
	SuccessfulTrades int64  `json:"successfulTrades"` // 成功交易次数
	FailedTrades     int64  `json:"failedTrades"`     // 失败交易次数
	IsBlacklisted    bool   `json:"isBlacklisted"`    // 是否被拉黑
}

// EnergyOffer 能源供应信息
type EnergyOffer struct {
	OfferID         string    `json:"offerId"`         // 供应ID
	SellerID        string    `json:"sellerId"`        // 卖方微网ID
	Amount          int64     `json:"amount"`          // 供应量（Wh）
	RemainingAmount int64     `json:"remainingAmount"` // 剩余量（Wh）
	Price           int64     `json:"price"`           // 单价（厘/Wh，1元/kWh=1厘/Wh）
	Status          string    `json:"status"`          // 状态：available, sold, cancelled
	CreatedTime     time.Time `json:"createdTime"`     // 创建时间
}

// EnergyDemand 能源需求信息
type EnergyDemand struct {
	DemandID        string    `json:"demandId"`        // 需求ID
	BuyerID         string    `json:"buyerId"`         // 买方微网ID
	Amount          int64     `json:"amount"`          // 需求量（Wh）
	RemainingAmount int64     `json:"remainingAmount"` // 剩余量（Wh）
	MaxPrice        int64     `json:"maxPrice"`        // 最高可接受价格（厘/Wh）
	Status          string    `json:"status"`          // 状态：open, fulfilled, cancelled
	CreatedTime     time.Time `json:"createdTime"`     // 创建时间
}

// Transaction 交易记录
type Transaction struct {
	TxID      string    `json:"txId"`      // 交易ID
	SellerID  string    `json:"sellerId"`  // 卖方微网ID
	BuyerID   string    `json:"buyerId"`   // 买方微网ID
	Amount    int64     `json:"amount"`    // 交易量（Wh）
	Price     int64     `json:"price"`     // 成交单价（厘/Wh）
	TotalCost int64     `json:"totalCost"` // 总价（分）
	Timestamp time.Time `json:"timestamp"` // 交易时间
	Status    string    `json:"status"`    // 状态：completed, failed
}

// InitLedger 初始化账本，创建3个微网账户
func (c *EnergyTradingContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	microgrids := []MicroGrid{
		{
			ID:               "MG1",
			Name:             "微网1",
			Balance:          1000000, // 10000元 = 1000000分
			LockedBalance:    0,
			EnergyBalance:    100000, // 100kWh = 100000Wh
			LockedEnergy:     0,
			Organization:     "MicroGrid1MSP",
			CreditScore:      InitialCreditScore,
			TotalTrades:      0,
			SuccessfulTrades: 0,
			FailedTrades:     0,
			IsBlacklisted:    false,
		},
		{
			ID:               "MG2",
			Name:             "微网2",
			Balance:          1000000,
			LockedBalance:    0,
			EnergyBalance:    50000, // 50kWh = 50000Wh
			LockedEnergy:     0,
			Organization:     "MicroGrid2MSP",
			CreditScore:      InitialCreditScore,
			TotalTrades:      0,
			SuccessfulTrades: 0,
			FailedTrades:     0,
			IsBlacklisted:    false,
		},
		{
			ID:               "MG3",
			Name:             "微网3",
			Balance:          1000000,
			LockedBalance:    0,
			EnergyBalance:    80000, // 80kWh = 80000Wh
			LockedEnergy:     0,
			Organization:     "MicroGrid3MSP",
			CreditScore:      InitialCreditScore,
			TotalTrades:      0,
			SuccessfulTrades: 0,
			FailedTrades:     0,
			IsBlacklisted:    false,
		},
	}

	for _, mg := range microgrids {
		mgJSON, err := json.Marshal(mg)
		if err != nil {
			return err
		}

		key, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{mg.ID})
		if err != nil {
			return fmt.Errorf("failed to create composite key: %v", err)
		}

		err = ctx.GetStub().PutState(key, mgJSON)
		if err != nil {
			return fmt.Errorf("failed to put microgrid to world state: %v", err)
		}
	}

	return nil
}

// RegisterMicroGrid 注册新的微网（仅管理员可调用，防止任意成员覆写账户）
func (c *EnergyTradingContract) RegisterMicroGrid(ctx contractapi.TransactionContextInterface, id string, name string, initialBalance int64, initialEnergy int64) error {
	// 权限控制: 账户注册/覆写是系统级操作
	if err := requireAdmin(ctx); err != nil {
		return err
	}
	// 获取调用者的组织信息
	orgID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("failed to get client identity: %v", err)
	}

	mg := MicroGrid{
		ID:               id,
		Name:             name,
		Balance:          initialBalance,
		LockedBalance:    0,
		EnergyBalance:    initialEnergy,
		LockedEnergy:     0,
		Organization:     orgID,
		CreditScore:      InitialCreditScore,
		TotalTrades:      0,
		SuccessfulTrades: 0,
		FailedTrades:     0,
		IsBlacklisted:    false,
	}

	mgJSON, err := json.Marshal(mg)
	if err != nil {
		return err
	}

	key, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{id})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}

	return ctx.GetStub().PutState(key, mgJSON)
}

// GetMicroGrid 查询微网信息
func (c *EnergyTradingContract) GetMicroGrid(ctx contractapi.TransactionContextInterface, id string) (*MicroGrid, error) {
	key, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{id})
	if err != nil {
		return nil, fmt.Errorf("failed to create composite key: %v", err)
	}

	mgJSON, err := ctx.GetStub().GetState(key)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if mgJSON == nil {
		return nil, fmt.Errorf("microgrid %s does not exist", id)
	}

	var mg MicroGrid
	err = json.Unmarshal(mgJSON, &mg)
	if err != nil {
		return nil, err
	}

	return &mg, nil
}

// CheckCreditAccess 检查微网信用分是否满足准入条件
func (c *EnergyTradingContract) CheckCreditAccess(mg *MicroGrid) error {
	// 检查是否被拉黑
	if mg.IsBlacklisted {
		return fmt.Errorf("microgrid %s is blacklisted and cannot participate in trading", mg.ID)
	}

	// 检查信用分是否达到准入阈值
	if mg.CreditScore < AccessThreshold {
		return fmt.Errorf("microgrid %s credit score (%d) is below access threshold (%d)", 
			mg.ID, mg.CreditScore, AccessThreshold)
	}

	return nil
}

// UpdateCreditScore 更新微网信用分
func (c *EnergyTradingContract) UpdateCreditScore(ctx contractapi.TransactionContextInterface, mgID string, isSuccess bool, penalty int) error {
	mg, err := c.GetMicroGrid(ctx, mgID)
	if err != nil {
		return err
	}

	// 根据交易结果更新信用分
	if isSuccess {
		// 交易成功：加分
		mg.CreditScore += SuccessReward
		if mg.CreditScore > MaxCreditScore {
			mg.CreditScore = MaxCreditScore
		}
		mg.SuccessfulTrades++
	} else {
		// 交易失败：扣分
		penaltyPoints := FailurePenalty
		if penalty > 0 {
			penaltyPoints = penalty
		}
		mg.CreditScore -= penaltyPoints
		if mg.CreditScore < MinCreditScore {
			mg.CreditScore = MinCreditScore
		}
		mg.FailedTrades++

		// 如果信用分低于准入阈值，自动拉黑
		if mg.CreditScore < AccessThreshold {
			mg.IsBlacklisted = true
		}
	}

	mg.TotalTrades++

	// 保存更新后的微网信息
	mgJSON, err := json.Marshal(mg)
	if err != nil {
		return err
	}

	key, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{mgID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}

	return ctx.GetStub().PutState(key, mgJSON)
}

// ResetCreditScore 重置微网信用分（管理员功能）
func (c *EnergyTradingContract) ResetCreditScore(ctx contractapi.TransactionContextInterface, mgID string) error {
	mg, err := c.GetMicroGrid(ctx, mgID)
	if err != nil {
		return err
	}

	mg.CreditScore = InitialCreditScore
	mg.IsBlacklisted = false

	mgJSON, err := json.Marshal(mg)
	if err != nil {
		return err
	}

	key, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{mgID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}

	return ctx.GetStub().PutState(key, mgJSON)
}

// GetCreditScore 查询微网信用分
func (c *EnergyTradingContract) GetCreditScore(ctx contractapi.TransactionContextInterface, mgID string) (int, error) {
	mg, err := c.GetMicroGrid(ctx, mgID)
	if err != nil {
		return 0, err
	}
	return mg.CreditScore, nil
}

// PublishEnergyOffer 发布能源供应
func (c *EnergyTradingContract) PublishEnergyOffer(ctx contractapi.TransactionContextInterface, offerID string, sellerID string, amount int64, price int64) error {
	// 验证卖方微网是否存在
	seller, err := c.GetMicroGrid(ctx, sellerID)
	if err != nil {
		return err
	}

	// ★ 信用检查：准入控制
	err = c.CheckCreditAccess(seller)
	if err != nil {
		return err
	}

	// 验证可用能源余额是否足够（余额 - 已锁定）
	availableEnergy := seller.EnergyBalance - seller.LockedEnergy
	if availableEnergy < amount {
		return fmt.Errorf("insufficient available energy: have %d Wh, need %d Wh", availableEnergy, amount)
	}

	// 锁定能源
	seller.LockedEnergy += amount
	sellerJSON, err := json.Marshal(seller)
	if err != nil {
		return err
	}
	sellerKey, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{sellerID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}
	err = ctx.GetStub().PutState(sellerKey, sellerJSON)
	if err != nil {
		return fmt.Errorf("failed to lock energy: %v", err)
	}

	// 获取交易时间戳
	txTimestamp, err := ctx.GetStub().GetTxTimestamp()
	if err != nil {
		return fmt.Errorf("failed to get tx timestamp: %v", err)
	}
	createdTime := time.Unix(txTimestamp.Seconds, int64(txTimestamp.Nanos))

	offer := EnergyOffer{
		OfferID:         offerID,
		SellerID:        sellerID,
		Amount:          amount,
		RemainingAmount: amount,
		Price:           price,
		Status:          "available",
		CreatedTime:     createdTime,
	}

	offerJSON, err := json.Marshal(offer)
	if err != nil {
		return err
	}

	offerKey, err := ctx.GetStub().CreateCompositeKey("Offer", []string{offerID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}

	return ctx.GetStub().PutState(offerKey, offerJSON)
}

// PublishEnergyDemand 发布能源需求
func (c *EnergyTradingContract) PublishEnergyDemand(ctx contractapi.TransactionContextInterface, demandID string, buyerID string, amount int64, maxPrice int64) error {
	// 验证买方微网是否存在
	buyer, err := c.GetMicroGrid(ctx, buyerID)
	if err != nil {
		return err
	}

	// ★ 信用检查：准入控制
	err = c.CheckCreditAccess(buyer)
	if err != nil {
		return err
	}

	// 计算需要锁定的最大资金（分） = amount(Wh) * maxPrice(厘/Wh) * 10
	maxTotalCost := (amount * maxPrice) * 10
	
	// 验证可用余额是否足够
	availableBalance := buyer.Balance - buyer.LockedBalance
	if availableBalance < maxTotalCost {
		return fmt.Errorf("insufficient available balance: have %d fen, need %d fen", availableBalance, maxTotalCost)
	}

	// 锁定资金
	buyer.LockedBalance += maxTotalCost
	buyerJSON, err := json.Marshal(buyer)
	if err != nil {
		return err
	}
	buyerKey, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{buyerID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}
	err = ctx.GetStub().PutState(buyerKey, buyerJSON)
	if err != nil {
		return fmt.Errorf("failed to lock balance: %v", err)
	}

	// 获取交易时间戳
	txTimestamp, err := ctx.GetStub().GetTxTimestamp()
	if err != nil {
		return fmt.Errorf("failed to get tx timestamp: %v", err)
	}
	createdTime := time.Unix(txTimestamp.Seconds, int64(txTimestamp.Nanos))

	demand := EnergyDemand{
		DemandID:        demandID,
		BuyerID:         buyerID,
		Amount:          amount,
		RemainingAmount: amount,
		MaxPrice:        maxPrice,
		Status:          "open",
		CreatedTime:     createdTime,
	}

	demandJSON, err := json.Marshal(demand)
	if err != nil {
		return err
	}

	demandKey, err := ctx.GetStub().CreateCompositeKey("Demand", []string{demandID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}

	return ctx.GetStub().PutState(demandKey, demandJSON)
}

// ExecuteTrade 执行交易 (简化版 - 核心逻辑优先)
func (c *EnergyTradingContract) ExecuteTrade(ctx contractapi.TransactionContextInterface, txID string, offerID string, demandID string) error {
	// ===== 第一步：读取数据 =====
	offerKey, _ := ctx.GetStub().CreateCompositeKey("Offer", []string{offerID})
	offerJSON, err := ctx.GetStub().GetState(offerKey)
	if err != nil || offerJSON == nil {
		return fmt.Errorf("offer not found: %s", offerID)
	}
	var offer EnergyOffer
	json.Unmarshal(offerJSON, &offer)
	if offer.Status != "available" {
		return fmt.Errorf("offer not available")
	}

	demandKey, _ := ctx.GetStub().CreateCompositeKey("Demand", []string{demandID})
	demandJSON, err := ctx.GetStub().GetState(demandKey)
	if err != nil || demandJSON == nil {
		return fmt.Errorf("demand not found: %s", demandID)
	}
	var demand EnergyDemand
	json.Unmarshal(demandJSON, &demand)
	if demand.Status != "open" {
		return fmt.Errorf("demand not open")
	}

	// ===== 第二步：验证条件 =====
	if offer.Price > demand.MaxPrice {
		return fmt.Errorf("price mismatch")
	}

	seller, _ := c.GetMicroGrid(ctx, offer.SellerID)
	buyer, _ := c.GetMicroGrid(ctx, demand.BuyerID)

	// 计算交易量和成本
	tradeAmount := offer.RemainingAmount
	if demand.RemainingAmount < tradeAmount {
		tradeAmount = demand.RemainingAmount
	}
	// 准确计算：总价(分) = amount(Wh) * price(厘/Wh) * 10
	totalCost := (tradeAmount * offer.Price) * 10

	// ===== 第三步：验证资源充足 =====
	if seller.EnergyBalance < tradeAmount {
		return fmt.Errorf("seller insufficient energy")
	}
	if buyer.Balance < totalCost {
		return fmt.Errorf("buyer insufficient balance")
	}

	// ===== 第四步：执行转账 (核心逻辑) =====
	// 卖方：能源留在EnergyBalance，现金到Balance
	seller.EnergyBalance -= tradeAmount
	seller.Balance += totalCost
	seller.LockedEnergy -= tradeAmount  // 释放锁定的能源
	seller.SuccessfulTrades++
	seller.TotalTrades++
	if seller.CreditScore < MaxCreditScore {
		seller.CreditScore++  // 交易成功+1分
	}
	
	// 买方：现金从Balance离开，能源进入EnergyBalance
	buyer.Balance -= totalCost
	buyer.EnergyBalance += tradeAmount
	buyer.LockedBalance -= totalCost  // 释放锁定的现金
	buyer.SuccessfulTrades++
	buyer.TotalTrades++
	if buyer.CreditScore < MaxCreditScore {
		buyer.CreditScore++  // 交易成功+1分
	}

	// ===== 第五步：保存账户状态 =====
	sellerKey, _ := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{seller.ID})
	sellerJSON, _ := json.Marshal(seller)
	if err := ctx.GetStub().PutState(sellerKey, sellerJSON); err != nil {
		return fmt.Errorf("failed to save seller: %v", err)
	}

	buyerKey, _ := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{buyer.ID})
	buyerJSON, _ := json.Marshal(buyer)
	if err := ctx.GetStub().PutState(buyerKey, buyerJSON); err != nil {
		return fmt.Errorf("failed to save buyer: %v", err)
	}

	// ===== 第六步：更新订单状态 =====
	offer.RemainingAmount -= tradeAmount
	if offer.RemainingAmount == 0 {
		offer.Status = "sold"
	}
	offerJSON, _ = json.Marshal(offer)
	ctx.GetStub().PutState(offerKey, offerJSON)

	demand.RemainingAmount -= tradeAmount
	if demand.RemainingAmount == 0 {
		demand.Status = "fulfilled"
	}
	demandJSON, _ = json.Marshal(demand)
	ctx.GetStub().PutState(demandKey, demandJSON)

	// ===== 第七步：记录交易 =====
	txTimestamp, _ := ctx.GetStub().GetTxTimestamp()
	tx := Transaction{
		TxID:      txID,
		SellerID:  offer.SellerID,
		BuyerID:   demand.BuyerID,
		Amount:    tradeAmount,
		Price:     offer.Price,
		TotalCost: totalCost,
		Timestamp: time.Unix(txTimestamp.Seconds, int64(txTimestamp.Nanos)),
		Status:    "completed",
	}
	txJSON, _ := json.Marshal(tx)
	txKey, _ := ctx.GetStub().CreateCompositeKey("Transaction", []string{txID})
	ctx.GetStub().PutState(txKey, txJSON)

	// ⭐ 注意：信用分更新由调用者单独调用，避免读写集冲突

	return nil
}

// GetTransaction 查询交易记录
func (c *EnergyTradingContract) GetTransaction(ctx contractapi.TransactionContextInterface, txID string) (*Transaction, error) {
	txKey, err := ctx.GetStub().CreateCompositeKey("Transaction", []string{txID})
	if err != nil {
		return nil, fmt.Errorf("failed to create composite key: %v", err)
	}

	txJSON, err := ctx.GetStub().GetState(txKey)
	if err != nil {
		return nil, fmt.Errorf("failed to read transaction: %v", err)
	}
	if txJSON == nil {
		return nil, fmt.Errorf("transaction %s does not exist", txID)
	}

	var tx Transaction
	err = json.Unmarshal(txJSON, &tx)
	if err != nil {
		return nil, err
	}

	return &tx, nil
}

// ReportDeviceFailure 报告设备故障（用于模拟设备故障场景）
func (c *EnergyTradingContract) ReportDeviceFailure(ctx contractapi.TransactionContextInterface, mgID string) error {
	// 扣除信用分（设备故障）
	return c.UpdateCreditScore(ctx, mgID, false, FailurePenalty)
}

// ReportViolation 报告严重违约（如数据造假、恶意行为等）
func (c *EnergyTradingContract) ReportViolation(ctx contractapi.TransactionContextInterface, mgID string, reason string) error {
	// 严重违约，重罚
	err := c.UpdateCreditScore(ctx, mgID, false, ViolationPenalty)
	if err != nil {
		return err
	}
	
	// 可以记录违约原因（这里简化处理）
	return nil
}

// UnblacklistMicroGrid 将微网从黑名单移除（管理员功能）
func (c *EnergyTradingContract) UnblacklistMicroGrid(ctx contractapi.TransactionContextInterface, mgID string) error {
	mg, err := c.GetMicroGrid(ctx, mgID)
	if err != nil {
		return err
	}

	mg.IsBlacklisted = false
	// 恢复到准入阈值
	if mg.CreditScore < AccessThreshold {
		mg.CreditScore = AccessThreshold
	}

	mgJSON, err := json.Marshal(mg)
	if err != nil {
		return err
	}

	key, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{mgID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}

	return ctx.GetStub().PutState(key, mgJSON)
}

// GetBlacklistedMicroGrids 查询所有被拉黑的微网
func (c *EnergyTradingContract) GetBlacklistedMicroGrids(ctx contractapi.TransactionContextInterface) ([]*MicroGrid, error) {
	allMicrogrids, err := c.GetAllMicroGrids(ctx)
	if err != nil {
		return nil, err
	}

	var blacklisted []*MicroGrid
	for _, mg := range allMicrogrids {
		if mg.IsBlacklisted {
			blacklisted = append(blacklisted, mg)
		}
	}

	return blacklisted, nil
}

// GetCreditStatistics 获取信用系统统计信息
func (c *EnergyTradingContract) GetCreditStatistics(ctx contractapi.TransactionContextInterface, mgID string) (map[string]interface{}, error) {
	mg, err := c.GetMicroGrid(ctx, mgID)
	if err != nil {
		return nil, err
	}

	successRate := 0.0
	if mg.TotalTrades > 0 {
		successRate = float64(mg.SuccessfulTrades) / float64(mg.TotalTrades) * 100
	}

	stats := map[string]interface{}{
		"microgridID":      mg.ID,
		"creditScore":      mg.CreditScore,
		"isBlacklisted":    mg.IsBlacklisted,
		"totalTrades":      mg.TotalTrades,
		"successfulTrades": mg.SuccessfulTrades,
		"failedTrades":     mg.FailedTrades,
		"successRate":      successRate,
		"canTrade":         !mg.IsBlacklisted && mg.CreditScore >= AccessThreshold,
	}

	return stats, nil
}

// CancelOffer 取消能源供应，释放锁定的能源
func (c *EnergyTradingContract) CancelOffer(ctx contractapi.TransactionContextInterface, offerID string) error {
	// 获取供应信息
	offerKey, err := ctx.GetStub().CreateCompositeKey("Offer", []string{offerID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}
	offerJSON, err := ctx.GetStub().GetState(offerKey)
	if err != nil {
		return fmt.Errorf("failed to read offer: %v", err)
	}
	if offerJSON == nil {
		return fmt.Errorf("offer %s does not exist", offerID)
	}

	var offer EnergyOffer
	err = json.Unmarshal(offerJSON, &offer)
	if err != nil {
		return err
	}

	// 只能取消可用状态的供应
	if offer.Status != "available" {
		return fmt.Errorf("cannot cancel offer with status: %s", offer.Status)
	}

	// 获取卖方信息
	seller, err := c.GetMicroGrid(ctx, offer.SellerID)
	if err != nil {
		return err
	}

	// 释放锁定的能源
	seller.LockedEnergy -= offer.RemainingAmount
	if seller.LockedEnergy < 0 {
		seller.LockedEnergy = 0 // 防止出现负数
	}

	// 更新卖方状态
	sellerJSON, err := json.Marshal(seller)
	if err != nil {
		return err
	}
	sellerKey, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{seller.ID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}
	err = ctx.GetStub().PutState(sellerKey, sellerJSON)
	if err != nil {
		return fmt.Errorf("failed to update seller: %v", err)
	}

	// 更新供应状态
	offer.Status = "cancelled"
	offerJSON, err = json.Marshal(offer)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(offerKey, offerJSON)
}

// CancelDemand 取消能源需求，释放锁定的资金
func (c *EnergyTradingContract) CancelDemand(ctx contractapi.TransactionContextInterface, demandID string) error {
	// 获取需求信息
	demandKey, err := ctx.GetStub().CreateCompositeKey("Demand", []string{demandID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}
	demandJSON, err := ctx.GetStub().GetState(demandKey)
	if err != nil {
		return fmt.Errorf("failed to read demand: %v", err)
	}
	if demandJSON == nil {
		return fmt.Errorf("demand %s does not exist", demandID)
	}

	var demand EnergyDemand
	err = json.Unmarshal(demandJSON, &demand)
	if err != nil {
		return err
	}

	// 只能取消开放状态的需求
	if demand.Status != "open" {
		return fmt.Errorf("cannot cancel demand with status: %s", demand.Status)
	}

	// 获取买方信息
	buyer, err := c.GetMicroGrid(ctx, demand.BuyerID)
	if err != nil {
		return err
	}

	// 计算需要释放的锁定资金
	lockedAmount := (demand.RemainingAmount * demand.MaxPrice) / 10
	buyer.LockedBalance -= lockedAmount
	if buyer.LockedBalance < 0 {
		buyer.LockedBalance = 0 // 防止出现负数
	}

	// 更新买方状态
	buyerJSON, err := json.Marshal(buyer)
	if err != nil {
		return err
	}
	buyerKey, err := ctx.GetStub().CreateCompositeKey("MicroGrid", []string{buyer.ID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}
	err = ctx.GetStub().PutState(buyerKey, buyerJSON)
	if err != nil {
		return fmt.Errorf("failed to update buyer: %v", err)
	}

	// 更新需求状态
	demand.Status = "cancelled"
	demandJSON, err = json.Marshal(demand)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(demandKey, demandJSON)
}

// GetAllMicroGrids 查询所有微网
func (c *EnergyTradingContract) GetAllMicroGrids(ctx contractapi.TransactionContextInterface) ([]*MicroGrid, error) {
	resultsIterator, err := ctx.GetStub().GetStateByPartialCompositeKey("MicroGrid", []string{})
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var microgrids []*MicroGrid
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var mg MicroGrid
		err = json.Unmarshal(queryResponse.Value, &mg)
		if err != nil {
			return nil, err
		}
		microgrids = append(microgrids, &mg)
	}

	return microgrids, nil
}

// GetAllOffers 查询所有供应信息
func (c *EnergyTradingContract) GetAllOffers(ctx contractapi.TransactionContextInterface) ([]*EnergyOffer, error) {
	resultsIterator, err := ctx.GetStub().GetStateByPartialCompositeKey("Offer", []string{})
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var offers []*EnergyOffer
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var offer EnergyOffer
		err = json.Unmarshal(queryResponse.Value, &offer)
		if err != nil {
			return nil, err
		}
		offers = append(offers, &offer)
	}

	return offers, nil
}

// GetAllTransactions 查询所有交易记录
func (c *EnergyTradingContract) GetAllTransactions(ctx contractapi.TransactionContextInterface) ([]*Transaction, error) {
	resultsIterator, err := ctx.GetStub().GetStateByPartialCompositeKey("Transaction", []string{})
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var transactions []*Transaction
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var tx Transaction
		err = json.Unmarshal(queryResponse.Value, &tx)
		if err != nil {
			return nil, err
		}
		transactions = append(transactions, &tx)
	}

	return transactions, nil
}
