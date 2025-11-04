# ECA-ResNet152

## 1. 模型概述

 ECA-ResNet152是一种结合了高效通道注意力机制（Efficient Channel Attention, ECA）的深度卷积神经网络模型，基于经典的ResNet152架构改进而来。ResNet152由微软研究院的Kaiming He等人于2016年提出，通过引入残差学习（Residual Learning）解决了深层网络中的梯度消失问题，成为深度学习领域的里程碑之一。ECA-ResNet152在ResNet152的基础上，融入了ECA模块，该模块通过轻量级的通道注意力机制，能够在不显著增加计算成本的情况下，自适应地增强重要特征通道的权重，从而提升模型的表征能力。

## 2. 快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

1. 虚拟环境启动：

    ```bash
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ECA-ResNet152
    
    conda activate torch_env
    
    # 执行以下命令验证环境是否正确，正确则会打印版本信息
    python -c 'import torch_sdaa'
    ```

2. 安装python依赖：

    ```bash
    pip install -r requirements.txt
    ```

### 2.2 准备数据集

ECA-ResNet152运行在ImageNet数据集上，数据集配置可以参考[https://blog.csdn.net/xzxg001/article/details/142465729](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fxzxg001%2Farticle%2Fdetails%2F142465729) 

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

 模型训练2h，得到结果如下 

| 加速卡数量 |     模型      | 混合精度 | Batch Size | Epoch | train_loss | AccTop1 |
| :--------: | :-----------: | :------: | :--------: | :---: | :--------: | :-----: |
|     1      | ECA-ResNet152 |   amp    |     64     |   3   |   4.485    |  0.226  |

