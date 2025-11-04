# MnasNet

## 1.模型概述

MnasNet是由 Google 在 2018 年提出的一种高效轻量级神经网络架构，专为移动设备设计。其核心创新在于结合自动化神经架构搜索（NAS） 和 多目标优化，在延迟（Latency）与准确率（Accuracy）之间实现最佳平衡。  

MnasNet 采用分层搜索空间，允许不同层使用不同的卷积操作（如深度可分离卷积、普通卷积），并优化模型在真实移动设备（如 Pixel Phone）上的推理速度，而非仅理论计算量（FLOPs）。其最终模型在 ImageNet 分类任务上达到 74.0% Top-1 准确率，同时保持较低的延迟（<80ms）。MnasNet 的设计理念影响了后续移动端模型（如 EfficientNet），成为轻量化网络的重要基准之一。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

1. 虚拟环境启动：

    ```bash
    cd <ModelZoo_path>/PyTorch/contrib/Classification/MnasNet
    
    conda activate torch_env
    
    # 执行以下命令验证环境是否正确，正确则会打印版本信息
    python -c 'import torch_sdaa'
    ```

2. 安装python依赖：

    ```bash
    pip install -r requirements.txt
    ```

### 2.2 准备数据集

MnasNet运行在ImageNet数据集上，数据集配置可以参考[https://blog.csdn.net/xzxg001/article/details/142465729](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fxzxg001%2Farticle%2Fdetails%2F142465729) 

### 2.3 启动训练

该模型支持单机单核组、单机单卡 

**单机单核组**

```
python train.py --dataset_path /data/datasets/imagenet --batch_size 128 --epochs 1 --distributed False --lr 0.01 --autocast True
```

**单机单卡**

```
torchrun --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet --batch_size 128 --epochs 1 --distributed True --lr 0.01 --autocast True
```

### 2.4 训练结果

模型训练2h，得到结果如下

| 加速卡数量 |  模型   | 混合精度 | Batch Size | Iterations | train_loss | AccTop1 |
| :--------: | :-----: | :------: | :--------: | :--------: | :--------: | :-----: |
|     1      | MnasNet |   amp    |    128     |    874     |   6.291    |    /    |

#### 情况说明：

在实际训练过程中，在2h的时间内，一个epoch中只能运行874个iteration，训练速度慢。