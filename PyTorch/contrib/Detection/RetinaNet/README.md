###  RetinaNet

**1.模型概述** 

RetinaNet是一种单阶段目标检测模型，由He等人提出，旨在解决传统单阶段检测器中正负样本不平衡的问题。该模型的核心创新是引入了Focal Loss损失函数，通过降低易分类背景样本的影响，提升模型对难分类前景物体的检测能力。RetinaNet的网络结构由主干网络（如ResNet）、特征金字塔网络（FPN）以及两个子网络（分类子网络和边框回归子网络）组成。它能够实现多尺度目标检测，并在自动驾驶、视频监控、工业质检等领域表现出色。

**2.快速开始**

使用本模型执行训练的主要流程如下：

基础环境安装：介绍训练前需要完成的基础环境检查和安装。

获取数据集：介绍如何获取训练所需的数据集。

启动训练：介绍如何运行训练。

**2.1 基础环境安装**

注意激活自身环境，然后可以直接运行下述代码进行安装或后续经requirements.txt统一安装

    pip install cython
    pip install pycocotools
    pip install matplotlib
（注意克隆torch.sdaa库）

**2.2 获取数据集**


coco数据集可以在官网进行下载；


**2.3 启动训练**

运行脚本在当前文件下，该模型在可以支持4卡分布式训练

1.cd到指定路径下，源码是一个比较大的框架（根据实际情况）


    cd ../references/detection
    pip install -r requirements.txt

2.运行指令

注意此时使用四卡，且指定了模型名称为Wide_ResNet101_2以及数据集的地址（数据集里面需要包括train和val文件）

**单机单核组**

   torchrun --nproc_per_node=1 train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1

**单机单卡**

    torchrun --nproc_per_node=4 train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1

**2.4 训练结果**

注：该代码不支持数据格式转换为NHWC（channels_last）
| 加速卡数量  | 模型  | 混合精度  | Batch_Size  | bbox_regression  |
|---|---|---|---|---|
| 1  | RetinaNet  | 是  |  2  |  0.6780 |



开始训练时loss有一些下降趋势，训练一段时间后loss值爆炸，bbox_regression指标值在训练过程中没有很大变化,classification指标值随loss的剧增而剧增
