###  regnet

**1.模型概述** 

RegNet 是一种由 Facebook AI Research（FAIR）团队提出的新型神经网络设计范式，旨在通过设计网络设计空间（Designing Network Design Spaces）来发现适用于各种环境的通用设计原则。RegNet 的核心思想是将网络的宽度和深度参数化为一个分段线性函数，从而生成一系列简单且规则的网络结构。与传统的神经架构搜索（NAS）方法不同，RegNet 不是专注于搜索单个最优网络实例，而是通过迭代缩减高维搜索空间，最终得到一个低维的、性能优秀的网络设计空间。
RegNet 的设计空间提供了简单而快速的网络模型，这些模型在不同计算资源限制下都能表现出色。例如，在可比的训练设置和计算量（FLOPs）下，RegNet 模型的性能优于流行的 EfficientNet 模型，并且在 GPU 上的运行速度提高了 5 倍。此外，RegNet 在多个计算机视觉任务中表现出色，包括图像分类、目标检测和语义分割等.

**2.快速开始**

使用本模型执行训练的主要流程如下：

基础环境安装：介绍训练前需要完成的基础环境检查和安装。

获取数据集：介绍如何获取训练所需的数据集。

启动训练：介绍如何运行训练。

**2.1 基础环境安装**

注意激活自身环境
（注意克隆torch.sdaa库）

**2.2 获取数据集**


Imagenet数据集可以在官网进行下载；


**2.3 启动训练**

运行脚本在当前文件下，该模型在可以支持4卡分布式训练

1.cd到指定路径下，源码是一个比较大的框架（根据实际情况）


    cd ../references/classification
    pip install -r requirements.txt

2.运行指令

注意此时使用四卡，且指定了模型名称为RegNet以及数据集的地址（数据集里面需要包括train和val文件）

**单机单核组**（cd 到指定路径下）

    cd ../references/classification/
    python train.py  --model regnet_x_400mf --world-size 1  --epochs 100 --batch-size 128 --wd 0.00005 --lr=0.8 --lr-scheduler=cosineannealinglr --lr-warmup-method=linear --lr-warmup-epochs=5  --lr-warmup-decay=0.1

**单机单卡**

    torchrun --nproc_per_node=4 train.py\
     --model regnet_x_400mf --epochs 100 --batch-size 128 --wd 0.00005 --lr=0.8\
     --lr-scheduler=cosineannealinglr --lr-warmup-method=linear\
     --lr-warmup-epochs=5 --lr-warmup-decay=0.1

**2.4 训练结果**
|加速卡数量| 模型  |  混合精度 | Batch_Size  |  Shape |  AccTop1 |
|---|---|---|---|---|---|
|  1 | RegNet  |  是 |  128 |  224x224  |  0.176 |
