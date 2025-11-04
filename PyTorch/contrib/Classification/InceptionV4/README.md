# InceptionV4

## 1.模型概述

 InceptionV4是Google团队在2016年提出的一种深度卷积神经网络模型，属于Inception系列的最新演进版本之一。Inception系列模型以其独特的Inception模块而闻名，该模块通过并行使用不同尺寸的卷积核和池化操作，能够高效地提取多尺度特征，从而提升模型的表现能力。InceptionV4在之前版本（如InceptionV1、InceptionV2和InceptionV3）的基础上，进一步优化了网络结构和训练策略，使其在图像分类和目标检测等计算机视觉任务中达到了更高的性能。 

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

1. 虚拟环境启动：

    ```bash
    cd <ModelZoo_path>/PyTorch/contrib/Classification/InceptionV4
    
    conda activate torch_env
    
    # 执行以下命令验证环境是否正确，正确则会打印版本信息
    python -c 'import torch_sdaa'
    ```

2. 安装python依赖：

    ```bash
    pip install -r requirements.txt
    ```

### 2.2 准备数据集

InceptionV4运行在ImageNet数据集上，数据集配置可以参考[https://blog.csdn.net/xzxg001/article/details/142465729](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fxzxg001%2Farticle%2Fdetails%2F142465729) 

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

模型训练2h，得到结果如下

| 加速卡数量 |    模型     | 混合精度 | Batch Size | Epoch | train_loss | AccTop1 |
| :--------: | :---------: | :------: | :--------: | :---: | :--------: | :-----: |
|     1      | InceptionV4 |   amp    |    128     |   5   |   3.823    |  0.336  |

