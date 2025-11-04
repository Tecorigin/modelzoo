###  unimp_arxiv

**1.模型概述** 

PyG （PyTorch Geometric）是一个建立在PyTorch之上的库，可以为与结构化数据相关的广泛应用轻松编写和训练图神经网络（gnn）。

**2.快速开始**

使用本模型执行训练的主要流程如下：

基础环境安装：介绍训练前需要完成的基础环境检查和安装。

获取数据集：介绍如何获取训练所需的数据集。

启动训练：介绍如何运行训练。

**2.1 基础环境安装**

注意激活自身环境
（注意克隆torch.sdaa库）

**2.2 获取数据集**

执行运行命令，自动下载数据集


**2.3 启动训练**

1.安装依赖

    pip install torch_geometric

2.运行指令

**单机单卡**
  cd examples
  export TORCH_SDAA_AUTOLOAD=cuda_migrate
  python unimp_arxiv.py
