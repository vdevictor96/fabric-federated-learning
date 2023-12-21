# Exit on error
set -e

# Config
# TODO add number of organisations and channels
chain_channel="federation_channel"
chaincode_name="federation"
chaincode_path="../chaincode/federation"

# Bring network up - single node Raft 
./network.sh down # remove any containers from previous runs (optional)
./network.sh up

# Create channel
./network.sh createChannel


# Deploy chaincode 
./network.sh deployCC \
    -c $chain_channel \
    -ccn $chaincode_name \
    -ccp $chaincode_path \
    -ccl typescript

export PATH=${PWD}/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/


# Environment variables for Org1
. scripts/envVar.sh
# Set environment variables for the peer org
setGlobals 1

# initialize the ledger
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls \
  --cafile "${PWD}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" \
  -C $chaincode_name \
  -n $chain_channel \
  --peerAddresses localhost:7051 \
  --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" \
  --peerAddresses localhost:9051 \
  --tlsRootCertFiles "${PWD}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" \
  -c '{"function":"InitLedger","Args":[]}'

