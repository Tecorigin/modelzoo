# ECA-MobileNetV2

## 1.模型概述

ECA-MobileNetV2是一种高效的卷积神经网络模型，由研究者们在MobileNetV2的基础上引入了高效通道注意力机制（Efficient Channel Attention, ECA）而提出。该模型在轻量级网络设计中取得了显著的进展，并在多个视觉任务中展现了优越的性能。ECA-MobileNetV2通过结合MobileNetV2的轻量级结构和ECA模块的通道注意力机制，能够在保持较低计算复杂度的同时，有效提升特征表示能力。这一创新使得ECA-MobileNetV2在移动设备和嵌入式系统等资源受限的场景中具有广泛的应用前景，并进一步推动了轻量级深度学习模型在计算机视觉领域的发展。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

1. 虚拟环境启动：

    ```bash
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ECA-MobileNetV2
    
    conda activate torch_env
    
    # 执行以下命令验证环境是否正确，正确则会打印版本信息
    python -c 'import torch_sdaa'
    ```

2. 安装python依赖：

    ```bash
    pip install -r requirements.txt
    ```

### 2.2 准备数据集

 ECA-MobileNetV2运行在ImageNet数据集上，数据集配置可以参考[https://blog.csdn.net/xzxg001/article/details/142465729](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fxzxg001%2Farticle%2Fdetails%2F142465729) 

### 2.3 启动训练

该模型支持单机单核组、单机单卡

**单机单核组**

```
python train.py --dataset_path /data/datasets/imagenet --batch_size 64 --epochs 20 --distributed False --lr 0.01 --autocast True
```

**单机单卡**

```
torchrun --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet --batch_size 64 --epochs 20 --distributed True --lr 0.01 --autocast True
```

### 2.4 训练结果

| 加速卡数量 |      模型       | 混合精度 | Batch Size | Iteration | train_loss | AccTop1 |
| :--------: | :-------------: | :------: | :--------: | :-------: | :--------: | :-----: |
|     1      | ECA-MobileNetV2 |    是    |     64     |   2092    |     /      |    /    |

#### 情况说明：

该模型训练速度过慢，在2h训练过程中，仅在第一个epoch中训练了2092个iteration，但是loss是下降的。