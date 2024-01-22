# Fabric Federated Learning

Master thesis\
M.Sc. Data Science and Artificial Intelligence (DSAI)\
Saarland University

Title: **Decentralised AI: A Blockchain-Based Federated Learning Implementation**\
Author: **Víctor Martínez Palomares**

Project done in collaboration with the **Max Planck Institute for Software Systems (MPI-SWS)**, supervised by Prof. Dr. Krishna P. Gummadi.

This project implements a Blockchain-Based Federated Learning framework using [Hyperledger Fabric](https://www.hyperledger.org/use/fabric) as permissioned blockchain and BERT-based models for binary text classification on depression (as a mental health condition) datasets.

## Index
- [Structure](#structure)
- [Getting Started](#getting-started)
  - [Blockchain Network](#blockchain-network)
  - [Chaincode](#chaincode)
  - [Gateway](#gateway)
  - [Client](#client)
- [Centralised Machine Learning](#centralised-machine-learning)
  - [Train](#train)
  - [Test](#test)
  - [Debug](#debug)
- [Centralised Federated Learning](#centralised-federated-learning)
- [Blockchain-Based Federated Learning](#blockchain-based-federated-learning)


## Structure 
- [`config/`](./config/) contains config files to deploy the blockchain network.
- [`federated-learning/`](./federated-learning/) contains all the logic for the federated learning, including the python ML code, the fabric smart contracts (chaincode) and the nodejs gateway.
- [`federated-learning/chaincode/`](./federated-learning/chaincode/) contains the TypeScript smart contract for the aggregation of the local models.
- [`federated-learning/client/`](./federated-learning/client/) contains the Python logic for training the BERT models and connecting to the gateway.
- [`federated-learning/gateway/`](./federated-learning/gateway/) contains the NestJS server that bridges the Python BERT models with the Hyperledger Fabric blockchain. 
- [`test-network/`](./test-network/) contains the blockchain test-network provided by [fabric-samples](https://github.com/hyperledger/fabric-samples), with added functionality to deploy the project's smart contracts.


## Getting Started

> ⚠️ Please note that these instructions are tailored to a Linux environment (Ubuntu 22.04 LTS) and may differ for other OS or Linux distributions.


### Blockchain Network

Install the Hyperledger Fabric [prerequisite software](https://hyperledger-fabric.readthedocs.io/en/release-2.5/prereqs.html#linux) to run a Docker-based Fabric test network on your local machine.

Bring up the blockchain test network with Docker:

```sh
cd test-network
./launchNetwork.sh 
```
This script will deploy a Fabric network with Docker . The [Hyperledger Fabric network](https://hyperledger-fabric.readthedocs.io/en/release-2.5/test_network.html) will consist of two peer organizations and an ordering organization with Raft ordering service.

It will also install the npm libraries required by the smart contracts (chaincode), specified in the package.json files, before compiling them and adding them to the blockchain nodes.

The network is meant to be used only as a tool for education and testing and not as a model for how to set up a network. To learn how to use Fabric in production, see [Deploying a production network](https://hyperledger-fabric.readthedocs.io/en/release-2.5/deployment_guide_overview.html).

### Chaincode


You can debug the chaincode with the help of [Hyperledger Fabric Debugger Plugin](https://github.com/spydra-tech/fabric-debugger) by Spydra.

Follow these steps:
1. Install the [Visual Studio Code extension](https://marketplace.visualstudio.com/items?itemName=Spydra.hyperledger-fabric-debugger).
2. Launch the Local Fabric Network provided in the extension menu.
3. Go to [test.fabric](federated-learning/chaincode/federation/test.fabric) file inside the [federation](federated-learning/chaincode/federation/) smart contract.
4. Create a request and invoke it. It will run in debug mode and trigger the added breakpoints.

### Gateway

Execute the following command to run the gateway in the local machine.

This command installs the required dependencies for the NestJS framework and the Fabric Gateway library.

```sh
cd federated-learning/gateway/
npm install 
```


To start the NestJS server in http://localhost:3000.
```sh
npm run start
```
To start in dev mode:
```sh
npm run start:dev
```

### Client

To execute the Python client you need to install the Python libraries in the requirements.txt file.

First, activate your python environment where you will be running the BERT model training. 

You can do that using conda with the following command:
```sh
conda activate your-existing-environment
```
or create a new environment:
```sh
conda create --name my-new-environment
conda activate my-new-environment
```

You can alternatively manage your environment with venv:
```sh
python3 -m venv my-new-environment
source my-new-environment/bin/activate
```

Now, install the requirements.txt libraries:
```sh
cd federated-learning/client
pip install -r requirements.txt
```


## Centralised Machine Learning
### Train
To run the fine-tuning of the BERT model in a centralised approach, execute the following command:

```sh
cd federated-learning
python client.run_train --config_file ./client/config/bert_tiny_config.json
```
You can train the model with your own configuration by creating a configuration file and passing it as an argument to --config-file.\
If you do not provide a valid configuration file the model will train with the [default-config](federated-learning/client/config/default_config.json) values.


To see the different configuration parameters and valid values run:
```sh
cd federated-learning
python client.run_train --show_config
```

### Test
To test the fine-tuned BERT model, execute the following command:
```sh
cd federated-learning
python client.run_test --config_file ./client/config/bert_tiny_test_config.json
```
You can test the model with your own configuration by creating a configuration file and passing it as an argument to --config-file.\
If you do not provide a valid configuration file the model will train with the [default_test_config](federated-learning/client/config/default_test_config.json) values.


To see the different configuration parameters and valid values run:
```sh
cd federated-learning
python client.run_test --show_config
```
### Debug
To execute the Jupyter Notebooks in Visual Studio Code you need to install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

To debug the code you can use the Jupyter Notebooks [train.ipynb](federated-learning/client/notebooks/train.ipynb) and [test.ipynb](federated-learning/client/notebooks/test.ipynb) and run the cells in debug mode.



## Centralised Federated Learning

## Blockchain-Based Federated Learning






