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
  - [Gateway](#gateway)
  - [Client](#client)
- [Train](#train)
- [Test](#test)
- [Debug](#debug)
  - [Chaincode](#chaincode)
- [Datasets](#datasets)



## Structure 
- [`config/`](./config/) contains config files to deploy the blockchain network.
- [`federated-learning/`](./federated-learning/) contains all the logic for the federated learning, including the python ML code, the fabric smart contracts (chaincode) and the nodejs gateway.
- [`federated-learning/chaincode/`](./federated-learning/chaincode/) contains the TypeScript smart contract for the aggregation of the local models.
- [`federated-learning/client/`](./federated-learning/client/) contains the Python logic for training the BERT models and connecting to the gateway.
- [`federated-learning/client/data/datasets`](./federated-learning/client/data/datasets) contains the public datasets used for the ablation studies.
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

This network is meant to be used only as a tool for education and testing and not as a model for how to set up a network. To learn how to use Fabric in production, see [Deploying a production network](https://hyperledger-fabric.readthedocs.io/en/release-2.5/deployment_guide_overview.html).

> ⚠️ If you are deploying the blockchain in a shared environment where docker runs in [rootless mode](https://docs.docker.com/engine/security/rootless/) you should set the `DOCKER_HOST` environment variable first `export DOCKER_HOST=unix:///run/user/ID/docker.sock` where ID is to be replace by the value of your `$UID` environment variable.



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

To execute the Python client you need to install the Python libraries in the [requirements.txt](ederated-learning/client/requirements.txt) file.

> ⚠️ This client is using python 3.11.7 and other version could case incompatibilities with the libraries versions used in [requirements.txt](ederated-learning/client/requirements.txt). More info in [client README file](ederated-learning/client/README.md)


First, activate your python environment where you will be running the BERT model training. 


You can do that using conda with the following command: [(Install conda following these instructions)](https://docs.anaconda.com/free/anaconda/install/linux/)
```sh
conda activate your-existing-environment
```
or create a new environment:
```sh
conda create --name my-new-environment python=3.11.7
conda activate my-new-environment
```

You can alternatively manage your environment with venv:
```shs
python3 -m venv my-new-environment python=3.11.7
source my-new-environment/bin/activate
```

Now, install the requirements.txt libraries:
```sh
cd federated-learning/client
conda install pip
pip install -r requirements.txt
```


## Train
To run the fine-tuning of the BERT model, execute the following command:

```sh
cd federated-learning
python client.run_train --config_file ./client/config/bert_tiny_config.json
```
or open the [train.ipynb](federated-learning/client/notebooks/train.ipynb) Jupyter Notebook on VSCode and run the last cell (it allows debugging of the code).


You can train the model with your own configuration by creating a configuration file and passing it as an argument to --config-file.\
The 'ml_mode' option can be set to three different values:
  - 'ml' for centralised machine learning
  - 'fl' for federated learning
  - 'bcfl' for blockchain-based federated learning (it needs the gateway running and the blockchain deployed)
Other configurable options available: 'model', 'dataset', 'layers', 'concurrency_flag', 'dp_epsilon', 'train_batch_size', etc.
If you do not provide a valid configuration file the model will train with the [default-config](federated-learning/client/config/default_config.json) values.


To see the different configuration parameters and valid values run:
```sh
cd federated-learning
python client.run_train --show_config
```
These are the default values (configurable):
```json
{
    "concurrency_flag": false,
    "data_distribution": "iid",
    "dataset": "twitter_dep",
    "device": "cuda",
    "dp_delta": 0.003,
    "dp_epsilon": 0.0,
    "eval_batch_size": 2,
    "eval_flag": true,
    "eval_size": 0.2,
    "fed_alg": "fedavg",
    "layers": 3,
    "learning_rate": 6e-05,
    "max_length": 512,
    "ml_mode": "ml",
    "model": "bert_tiny",
    "model_name": "bert_tiny",
    "models_path": "client/models/bert_tiny",
    "mu": 0.5,
    "num_clients": 5,
    "num_epochs": 10,
    "num_rounds": 10,
    "optimizer": "AdamW",
    "progress_bar_flag": true,
    "save_model": true,
    "scheduler": "linear",
    "scheduler_warmup_steps": 0,
    "seed": 200,
    "test_flag": false,
    "train_batch_size": 4,
    "train_size": 0.8
}
```

## Test
To test the fine-tuned BERT model, execute the following command:
```sh
cd federated-learning
python client.run_test --config_file ./client/config/bert_tiny_test_config.json
```
or open the [test.ipynb](federated-learning/client/notebooks/test.ipynb) Jupyter Notebook on VSCode and run the last cell (it allows debugging of the code).


You can test the model with your own configuration by creating a configuration file and passing it as an argument to --config-file.\
If you do not provide a valid configuration file the model will train with the [default_test_config](federated-learning/client/config/default_test_config.json) values.


To see the different configuration parameters and valid values run:
```sh
cd federated-learning
python client.run_test --show_config
```
These are the default values (configurable):
```json
{
    "dataset": "twitter_dep",
    "device": "cuda",
    "max_length": 512,
    "model": "bert_tiny",
    "model_path": "client/models/bert_tiny/bert_tiny_best.ckpt",
    "progress_bar_flag": true,
    "seed": 200,
    "test_batch_size": 8
}
```

## Debug

To debug the code you can use the Jupyter Notebooks [train.ipynb](federated-learning/client/notebooks/train.ipynb) and [test.ipynb](federated-learning/client/notebooks/test.ipynb) and run the cells in debug mode.

To execute the Jupyter Notebooks in Visual Studio Code you need to install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).


### Chaincode

You can debug the chaincode with the help of [Hyperledger Fabric Debugger Plugin](https://github.com/spydra-tech/fabric-debugger) by Spydra.

Follow these steps:
1. Install the [Visual Studio Code extension](https://marketplace.visualstudio.com/items?itemName=Spydra.hyperledger-fabric-debugger).
2. Launch the Local Fabric Network provided in the extension menu.
3. Go to [test.fabric](federated-learning/chaincode/federation/test.fabric) file inside the [federation](federated-learning/chaincode/federation/) smart contract.
4. Create a request and invoke it. It will run in debug mode and trigger the added breakpoints.

## Datasets

The datasets used in this project for fine-tuning the BERT Models are public datasets derived from web scrapping posts and comments from mental health, depression, and self-harm subreddits on [Reddit](https://www.reddit.com/) and tweets on [Twitter](https://twitter.com/).

Some dataset labels are validated using small datasets (500 posts) labelled by psychologists from the depression subreddit. Some others are labelled from users self-reporting their diagnosis of depression.

We fine-tuned with five different datasets:
- [twitter_dep](federated-learning/client/data/datasets/twitter_dep/twitter_dep_full.csv) from [Benchmarking Differential Privacy and Federated Learning for BERT Models](https://arxiv.org/abs/2106.13973) contains more than 3000 tweets labelleds as depression or non-depression.

- [acl_dep_sad](federated-learning/client/data/datasets/acl_dep_sad/acl_dep_sad_full.csv) from [Interpretability of Fine-grained Classification of Sadness and Depression](https://arxiv.org/abs/2203.10432) is a collection of six different datasets, three classifying depression and three classifying sadness. It contains 3256 samples of which 1914 samples were labelled as "sadness" and 1342 samples under the label "depression".

- [mixed_depression](federated-learning/client/data/datasets/mixed_depression/mixed_depression_full.csv) from [Identifying depression on Reddit: the effect of training data](https://aclanthology.org/W18-5903.pdf) contains almost 3000 posts collected from Reddit, from a large subreddit that is devoted to depression.

- [dreaddit](federated-learning/client/data/datasets/dreaddit/dreaddit_train.csv) from [Dreaddit: A Reddit Dataset for Stress Analysis in Social Media](https://arxiv.org/pdf/1911.00133.pdf) contains more than 3000 posts in Reddit from ten different subreddits — in the five domains of abuse, social, anxiety, PTSD, and financial — classified as depression vs. non-depression.

- deptweet from [DEPTWEET: A typology for social media texts to detect depression severities](https://arxiv.org/pdf/2210.05372.pdf) contains more than 40000 tweets labelled depending on their depression severity ranging 0-3. **This dataset is not available to the public. It is used under an user agreement contract for research purposes and hence not shared in this public repository.**






