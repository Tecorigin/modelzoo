### vit_pytorch

**1. 模型概述**

ViT 是一种基于 Transformer 架构的视觉模型，它在图像分类任务上取得了显著的成果


**2. 快速开始**

使用本模型执行训练的主要流程如下：
1. 基础环境按照：介绍训练前需要完成的基础环境检查和安装
2. 获取数据集：介绍如何获取训练所需的数据集
3. 启动训练：介绍如何运行训练

**2.1 基础环境安装**

注意激活自身环境
（注意克隆torch.sdaa库）

**2.2 获取数据集**

猫狗数据集，可于以下链接kaggle上下载

* Dogs vs. Cats Redux: Kernels Edition - https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition


**2.3 启动训练**

1. 安装依赖

    pip install vit-pytorch

2. 运行指令

    **单机单卡**
    cd examples
    python ./cats_and_dogs.py