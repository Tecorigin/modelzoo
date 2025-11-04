###  Wide_ResNet_2

**1.模型概述** 

Wide_ResNet_2 是一种基于宽残差网络（Wide Residual Networks）的深度学习架构，它通过增加网络的宽度（即通道数）而不是深度来提升模型的性能。与传统的 ResNet 相比，Wide_ResNet 在每一层的通道数上乘以一个扩展因子（如 2、4、8 等），从而显著增强模型的特征提取能力。此外，Wide_ResNet 通常还会结合 Dropout 技术来缓解过拟合问题。

**2.快速开始**

使用本模型执行训练的主要流程如下：

基础环境安装：介绍训练前需要完成的基础环境检查和安装。

获取数据集：介绍如何获取训练所需的数据集。

启动训练：介绍如何运行训练。

**2.1 基础环境安装**

注意激活自身环境，然后可以直接运行下述代码进行安装


    pip install -r requirements.txt
（注意克隆torch.sdaa库）

**2.2 获取数据集**


Imagenet数据集可以在官网进行下载；


**2.3 启动训练**

运行脚本在当前文件下，该模型在可以支持4卡分布式训练

1.cd到指定路径下，源码是一个比较大的框架（根据实际情况）


    cd ../references/classification

2.运行指令

注意此时使用四卡，且指定了模型名称为Wide_ResNet101_2以及数据集的地址（数据集里面需要包括train和val文件）

 **单机单核组**

    nohup torchrun --nproc_per_node=1 train.py --model Wide_ResNet101_2 --data-path /data/datasets/imagenet > wide_resnet_2.log 2>&1 &

**单机单卡**

    nohup torchrun --nproc_per_node=4 train.py --model Wide_ResNet101_2 --data-path /data/datasets/imagenet > wide_resnet_2.log 2>&1 &

**2.4 训练结果**
|加速卡数量| 模型  |  混合精度 | Batch_Size  |  Shape |  AccTop1 |
|---|---|---|---|---|---|
|  1 | Wide_ResNet_2  |  是 |  32 |  224*224  |  0.263 |
