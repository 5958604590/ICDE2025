# Cost-Efficient and Scalable Distributed Training for Graph Neural Networks

## Overview

This repository contains the implementation of CATGNN, a cost-efficient and scalable distributed training system for graph neural networks (GNNs).


## Requirements

<!--PyTorch v2.0.1-->
<!--DGL=1.1.0-->
<!--CUDA=11.8-->

[![](https://img.shields.io/badge/PyTorch-2.0.1-blueviolet)](https://pytorch.org/get-started/)
[![](https://img.shields.io/badge/DGL-1.1.0-blue)](https://www.dgl.ai/pages/start.html)
[![](https://img.shields.io/badge/CUDA-11.8-green)](https://developer.nvidia.com/cuda-11-8-0-download-archive)

GPU versions of [PyTorch](https://pytorch.org/get-started/) and [DGL](https://www.dgl.ai/pages/start.html) are required to run CATGNN. Please check the corresponding official websites for installation.


## System
![Image text](https://github.com/5958604590/ICDE2025/blob/main/System.png)


## Dataset
We use 16 real-world graph datasets that are downloaded from [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT), [DGL](https://www.dgl.ai/), and [OGB](https://ogb.stanford.edu/), and converted to the formats used by our system.
> 1. edge_list.csv: edge list of the graph.
> 2. feats.npy: node features (NumPy array).
> 3. class_map.json: node labels (dictionary).
> 4. role.json: role of Train/Val/Test split (dictionary).


## Running the code

We provide an example in 'example.py'. Follow the command below to run the code.

```bash
$ python3 example.py
```

Please refer to our paper for more details.
