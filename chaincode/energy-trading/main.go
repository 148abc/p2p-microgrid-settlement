package main

import (
	"log"
	"os"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

func main() {
	energyContract := new(EnergyTradingContract)

	chaincode, err := contractapi.NewChaincode(energyContract)
	if err != nil {
		log.Panicf("Error creating energy trading chaincode: %v", err)
	}

	// Chaincode-as-a-Service: server mode (peer connects to us) when
	// CHAINCODE_SERVER_ADDRESS is set; otherwise legacy client mode.
	if addr := os.Getenv("CHAINCODE_SERVER_ADDRESS"); addr != "" {
		server := &shim.ChaincodeServer{
			CCID:    os.Getenv("CORE_CHAINCODE_ID_NAME"),
			Address: addr,
			CC:      chaincode,
			TLSProps: shim.TLSProperties{
				Disabled: true,
			},
		}
		if err := server.Start(); err != nil {
			log.Panicf("Error starting energy trading chaincode server: %v", err)
		}
		return
	}

	if err := chaincode.Start(); err != nil {
		log.Panicf("Error starting energy trading chaincode: %v", err)
	}
}
