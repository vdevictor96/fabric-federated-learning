// import { Contract, Gateway, GatewayOptions, Network } from "fabric-network";
import {
  connect,
  Contract,
  Identity,
  Signer,
  signers,
  Gateway,
  Network,
  ConnectOptions,
} from "@hyperledger/fabric-gateway";
import { Client, credentials } from "@grpc/grpc-js";
import { promises as fs } from "fs";
import * as path from "path";
import { createPrivateKey } from "crypto";

export class GatewayClient {
  static readonly CHANNEL_NAME: string = GatewayClient.envOrDefault(
    "CHANNEL_NAME",
    "federationchannel"
  );
  static readonly CHAINCODE_NAME: string = GatewayClient.envOrDefault(
    "CHAINCODE_NAME",
    "federation"
  );
  static readonly MSP_ID: string = GatewayClient.envOrDefault(
    "MSP_ID",
    "Org1MSP"
  );
  static readonly CRYPTO_PATH: string = GatewayClient.envOrDefault(
    "CRYPTO_PATH",
    path.resolve(
      __dirname,
      "..",
      "..",
      "..",
      "test-network",
      "organizations",
      "peerOrganizations",
      "org1.example.com"
    )
  );
  static readonly KEY_DIRECTORY_PATH: string = GatewayClient.envOrDefault(
    "KEY_DIRECTORY_PATH",
    path.resolve(
      GatewayClient.CRYPTO_PATH,
      "users",
      "User1@org1.example.com",
      "msp",
      "keystore"
    )
  );
  static readonly CERT_PATH: string = GatewayClient.envOrDefault(
    "CERT_PATH",
    path.resolve(
      GatewayClient.CRYPTO_PATH,
      "users",
      "User1@org1.example.com",
      "msp",
      "signcerts",
      "cert.pem"
    )
  );
  static readonly TLS_CERT_PATH: string = GatewayClient.envOrDefault(
    "TLS_CERT_PATH",
    path.resolve(
      GatewayClient.CRYPTO_PATH,
      "peers",
      "peer0.org1.example.com",
      "tls",
      "ca.crt"
    )
  );
  static readonly PEER_ENDPOINT: string = GatewayClient.envOrDefault(
    "PEER_ENDPOINT",
    "localhost:7051"
  );
  static readonly PEER_HOST_ALIAS: string = GatewayClient.envOrDefault(
    "PEER_HOST_ALIAS",
    "peer0.org1.example.com"
  );
  static readonly UTF8_DECODER = new TextDecoder();
  static readonly ASSET_ID = `asset${Date.now()}`;

  private client: Client | undefined;
  private connectionOptions: ConnectOptions | undefined;
  private gateway: Gateway | undefined;

  private networkName: string | undefined;
  // private lastNetworkName: string | undefined;
  private _network: Network | undefined;

  private chaincodeName: string | undefined;
  // private lastChaincodeName: string | undefined;
  private _contract: Contract | undefined;

  constructor() {
    this.connect();
  }
  public async connect() {
    await this.displayInputParameters();

    // The gRPC client connection should be shared by all Gateway connections to this endpoint.
    if (!this.client) this.client = await this.newGrpcConnection();

    if (!this.gateway) {
      this.gateway = connect({
        client: this.client,
        identity: await this.newIdentity(),
        signer: await this.newSigner(),
        // Default timeouts for different gRPC calls
        evaluateOptions: () => {
          return { deadline: Date.now() + 5000 }; // 5 seconds
        },
        endorseOptions: () => {
          return { deadline: Date.now() + 15000 }; // 15 seconds
        },
        submitOptions: () => {
          return { deadline: Date.now() + 5000 }; // 5 seconds
        },
        commitStatusOptions: () => {
          return { deadline: Date.now() + 60000 }; // 1 minute
        },
      });
    }
    // if (!this.gateway) this.gateway = new Gateway();
    // this.gatewayOptions = gatewayOptions;

    // await this.gateway?.connect(connectionProfile, this.gatewayOptions);

    // return this;
  }

  //   public async getNetwork(): Promise<Network | undefined> {
  //     if (!this.networkName) return;

  //     this._network =
  //       this.lastNetworkName === this.networkName
  //         ? this._network
  //         : await this.gateway?.getNetwork(this.networkName);
  //     this.lastNetworkName = this.networkName;

  //     return this._network;
  //   }

  public async getNetwork(): Promise<Network | undefined> {
    if (!this.networkName) return;
    this._network = await this.gateway?.getNetwork(this.networkName);
    return this._network;
  }

  //   public async getContract(): Promise<Contract | undefined> {
  //     if (!this.networkName || !this.chaincodeName) return;

  //     const network = await this.getNetwork();
  //     this._contract =
  //       this.lastChaincodeName === this.chaincodeName
  //         ? this._contract
  //         : network?.getContract(this.chaincodeName);
  //     this.lastChaincodeName = this.chaincodeName;

  //     return this._contract;
  //   }

  public async getContract(): Promise<Contract | undefined> {
    if (!this.networkName || !this.chaincodeName) return;
    if (!this._network) await this.getNetwork();
    if (!this._network) return;
    this._contract = this._network?.getContract(this.chaincodeName);
    return this._contract;
  }

  public async submitTransaction(
    name: string,
    args: any[]
  ): Promise<Uint8Array | undefined> {
    if (!this._contract) await this.getContract();
    return await this._contract?.submitTransaction(name, ...args);
  }

  public async evaluateTransaction(
    name: string,
    args: any[]
  ): Promise<Uint8Array | undefined> {
    if (!this._contract) await this.getContract();
    return await this._contract?.evaluateTransaction(name, ...args);
  }

  // public network(networkName: string): Client {
  //   this.networkName = networkName;

  //   return this;
  // }

  // public contract(chaincodeName: string): Client {
  //   this.chaincodeName = chaincodeName;

  //   return this;
  // }

  public disconnect() {
    if (this.gateway) this.gateway.close();
    if (this.client) this.client.close();
  }

  /**
   * Creates a new gRPC connection to the Hyperledger Fabric network.
   * @returns A Promise resolving to a gRPC client.
   */
  private async newGrpcConnection(): Promise<Client> {
    const tlsRootCert = await fs.readFile(GatewayClient.TLS_CERT_PATH);
    const tlsCredentials = credentials.createSsl(tlsRootCert);
    return new Client(GatewayClient.PEER_ENDPOINT, tlsCredentials, {
      "grpc.ssl_target_name_override": GatewayClient.PEER_HOST_ALIAS,
    });
  }

  /**
   * Generates a new identity for connecting to the network.
   * @returns A Promise resolving to an Identity object.
   */
  private async newIdentity(): Promise<Identity> {
    const credentials = await fs.readFile(GatewayClient.CERT_PATH);
    return { mspId: GatewayClient.MSP_ID, credentials };
  }

  /**
   * Generates a new signer for signing transactions.
   * @returns A Promise resolving to a Signer object.
   */
  private async newSigner(): Promise<Signer> {
    const files = await fs.readdir(GatewayClient.KEY_DIRECTORY_PATH);
    const keyPath = path.resolve(GatewayClient.KEY_DIRECTORY_PATH, files[0]);
    const privateKeyPem = await fs.readFile(keyPath);
    const privateKey = createPrivateKey(privateKeyPem);
    return signers.newPrivateKeySigner(privateKey);
  }

  /**
   * displayInputParameters() will print the global scope parameters used by the main driver routine.
   */
  private async displayInputParameters(): Promise<void> {
    console.log(`channelName:       ${GatewayClient.CHANNEL_NAME}`);
    console.log(`chaincodeName:     ${GatewayClient.CHAINCODE_NAME}`);
    console.log(`mspId:             ${GatewayClient.MSP_ID}`);
    console.log(`cryptoPath:        ${GatewayClient.CRYPTO_PATH}`);
    console.log(`keyDirectoryPath:  ${GatewayClient.KEY_DIRECTORY_PATH}`);
    console.log(`certPath:          ${GatewayClient.CERT_PATH}`);
    console.log(`tlsCertPath:       ${GatewayClient.TLS_CERT_PATH}`);
    console.log(`peerEndpoint:      ${GatewayClient.PEER_ENDPOINT}`);
    console.log(`peerHostAlias:     ${GatewayClient.PEER_HOST_ALIAS}`);
  }

  /**
   * envOrDefault() will return the value of an environment variable, or a default value if the variable is undefined.
   */
  private static envOrDefault(key: string, defaultValue: string): string {
    return process.env[key] || defaultValue;
  }
}
