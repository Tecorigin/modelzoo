# FCOS

## 1.模型概述

FCOS（Fully Convolutional One-Stage Object Detection）是一种基于全卷积网络的一阶段目标检测模型，由Tian等人于2019年提出。与传统的两阶段目标检测模型（如Faster R-CNN）和基于锚框（Anchor-based）的一阶段模型（如YOLO和SSD）不同，FCOS摒弃了锚框的设计，采用了一种无锚框（Anchor-free）的检测方法，直接对特征图上的每个像素点进行目标边界框的预测。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

1. 虚拟环境启动：

    ```bash
    cd <ModelZoo_path>/PyTorch/contrib/Detection/FCOS
    
    conda activate torch_env
    
    # 执行以下命令验证环境是否正确，正确则会打印版本信息
    python -c 'import torch_sdaa'
    ```

2. 安装python依赖：

    ```bash
    pip install -r requirements.txt
    ```

### 2.2 准备数据集

FCOS运行在COCO数据集上，数据集配置可以参考 https://blog.csdn.net/HackerTom/article/details/117001560

### 2.3 启动训练

该模型支持单机单核组、单机单卡 

**单机单核组**

```
SDAA_LAUNCH_BLOCKING=1 python train.py --dataset_path /data/datasets/20241122/coco/train2017 --annotation_path /data/datasets/20241122/coco/annotations/instances_train2017.json --batch_size 4 --epochs 1 --distributed False --lr 0.01 --autocast True
```

**单机单卡**

```
 torchrun --nproc_per_node=4 train.py --dataset_path /data/datasets/20241122/coco/train2017 --annotation_path /data/datasets/20241122/coco/annotations/instances_train2017.json --batch_size 4 --epochs 1 --distributed True --lr 0.01 --autocast True
```

### 2.4 训练结果

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Iteration | train_loss | AccTop1 |
| :--------: | :--: | :------: | :--------: | :-------: | :--------: | :-----: |
|     1      | FCOS |   amp    |     4      |   6242    |     /      |    /    |

#### 情况说明：

该模型训练速度过慢，在2h训练过程中，仅在第一个epoch中训练了6242个iteration，但是loss是下降的。

