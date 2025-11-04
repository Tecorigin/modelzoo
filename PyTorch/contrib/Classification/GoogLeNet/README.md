### GoogleNet
### 1.模型概述
GoogLeNet 是 Google 在 2014 年提出的深度卷积神经网络（CNN），用于 ImageNet 图像分类任务，并赢得了 ILSVRC 2014 冠军。其核心创新是 Inception 模块，通过多尺度卷积（1×1、3×3、5×5）并行计算 提升特征提取能力，同时降低计算量。相比传统 CNN，GoogLeNet 更深（22 层）但计算效率更高，并引入了 1×1 卷积 进行降维，减少参数和计算开销。此外，它使用 辅助分类器（Auxiliary Classifiers）进行中间层监督，提高训练稳定性。GoogLeNet 开创了深度神经网络的高效设计，对后续模型影响深远。
### 2.快速开始
使用本模型训练流程如下：
1.基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2，获取数据集：介绍如何获取训练所需的数据集。
3.启动训练：介绍如何运行训练。
### 2.1基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。
### 2.2准备数据集
Googlenet在ImageNet数据集上运行
### 启动训练
该模型支持单机单核组和单机单卡
 **单机单核组** 

```
python -m torch.distributed.launch --nproc_per_node=1 train.py --dataset_path /root/data/bupt/datasets/imagenet \
--batch_size 32 --epochs 10 --distributed True --lr 0.0001 --autocast True
```
 **单机单卡** 

```
python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /root/data/bupt/datasets/imagenet \
--batch_size 32 --epochs 10 --distributed True --lr 0.0001 --autocast True
```
### 2.4训练结果
| 模型         | googlenet |
|------------|-----------|
| 加速卡        | 4         |
| batch size | 32        |
| epoch      | 10        |
| 混合精度       | 是         |
| loss       | 2.4536    |








