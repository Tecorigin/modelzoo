# Xception

## 1.模型概述

Xception是一种深度卷积神经网络模型，由Google的研究者François Chollet在2017年提出。Xception是对Inception架构的进一步扩展和改进，其核心思想是通过深度可分离卷积（Depthwise Separable Convolution来替代传统的卷积操作，从而在减少计算量的同时提升模型的性能。Xception的名称来源于“Extreme Inception”，表明它是Inception架构的一种极端形式。 

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

1. 虚拟环境启动：

    ```bash
    cd <ModelZoo_path>/PyTorch/contrib/Classification/Xception
    
    conda activate torch_env
    
    # 执行以下命令验证环境是否正确，正确则会打印版本信息
    python -c 'import torch_sdaa'
    ```

2. 安装python依赖：

    ```bash
    pip install -r requirements.txt
    ```

### 2.2 准备数据集

Xception运行在ImageNet数据集上，数据集配置可以参考[https://blog.csdn.net/xzxg001/article/details/142465729](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fxzxg001%2Farticle%2Fdetails%2F142465729) 

### 2.3 启动训练

该模型支持单机单核组、单机单卡 

**单机单核组**

```
python train.py --dataset_path /data/datasets/imagenet --batch_size 128 --epochs 20 --distributed False --lr 0.01 --autocast True
```

**单机单卡**

```
torchrun --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet --batch_size 128 --epochs 20 --distributed True --lr 0.01 --autocast True
```

### 2.4 训练结果

| 加速卡数量 |   模型   | 混合精度 | Batch Size | Epoch | train_loss | AccTop1 |
| :--------: | :------: | :------: | :--------: | :---: | :--------: | :-----: |
|     1      | Xception |   amp    |    128     |   3   |   4.297    |  0.257  |
