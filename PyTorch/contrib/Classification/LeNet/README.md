### LeNet
### 1.模型概述
LeNet 是 Yann LeCun 在 1989 年提出的早期卷积神经网络（CNN），主要用于 手写数字识别（如 MNIST 数据集）。它由 两层卷积层（C1、C3）、两层池化层（S2、S4）、两层全连接层（F5、F6）和最终的输出层 组成，采用 5×5 卷积核 进行特征提取，并通过 平均池化（Subsampling） 降维，减少计算量。
### 2.快速开始
使用本模型训练流程如下：
1.基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2，获取数据集：介绍如何获取训练所需的数据集。
3.启动训练：介绍如何运行训练。
### 2.1基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。
### 2.2准备数据集
Lenet在ImageNet数据集上运行
### 2.3启动训练
该模型支持单机单核组和单机单卡
 **单机单核组** 

```
MASTER_PORT=29501 python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 29501 train.py \
    --dataset_path /root/data/bupt/datasets/imagenet \
    --batch_size 32 --epochs 10 --distributed True --lr 0.0001 --autocast True
```
 **单机单卡** 

```
MASTER_PORT=29501 python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 29501 train.py \
    --dataset_path /root/data/bupt/datasets/imagenet \
    --batch_size 32 --epochs 10 --distributed True --lr 0.0001 --autocast True
```
### 2.4训练结果
| 模型         | lenet  |
|------------|--------|
| epoch      | 10     |
| batch size | 32     |
| 混合精度       | 是      |
| 加速卡数量      | 4      |
| loss       | 2.5522 |



