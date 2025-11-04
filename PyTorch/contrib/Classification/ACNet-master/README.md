###  ACNet 


**1.模型概述** 

ACNet是一种用于增强卷积神经网络（CNN）性能的开源项目，其核心是通过非对称卷积块（Asymmetric Convolution Blocks, ACB）来提升模型的特征提取能力和准确率。ACB由三个平行的卷积层组成，分别具有方形、水平和垂直的卷积核。通过将ACB替换传统CNN中的方形卷积核，ACNet能够在训练后将这些卷积核的输出相加，从而增强卷积核的骨架部分，提升模型对旋转和翻转图像的鲁棒性。此外，ACNet在不增加额外推理时间的情况下，显著提高了模型的准确率。它还具有良好的兼容性，可以方便地集成到现有的CNN架构中，如ResNet和DenseNe

**2.快速开始**

使用本模型执行训练的主要流程如下：

基础环境安装：介绍训练前需要完成的基础环境检查和安装。

获取数据集：介绍如何获取训练所需的数据集。

启动训练：介绍如何运行训练。

**2.1 基础环境安装**

注意激活自身环境

克隆torch_sdaa环境，激活环境

    

**2.2 获取数据集**

Imagenet数据集可以在官网进行下载；



**2.3 启动训练**


运行脚本在当前文件下，该模型在可以支持4卡分布式训练


1.进入当前目录

    cd ..
    此外，需要额外下载如下库
    pip install -r requirements.txt

2.为imagenet数据集创建一个软链接，imagenet文件下需要包括train和val文件夹


    ln -s /data/datasets/imagenet imagenet_data

3.设置环境变量，并设置为4卡运行

    export PYTHONPATH=.
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    
4.在ImageNet上训练一个带有非对称卷积块的ResNet-18模型

**单机单核组**

    nohup python -m torch.distributed.launch --nproc_per_node=1 acnet/do_acnet.py -a sres18 -b acb >>acnet.txt 2>&1 &

**单机单卡**

    nohup python -m torch.distributed.launch --nproc_per_node=4 acnet/do_acnet.py -a sres18 -b acb >>acnet.txt 2>&1 &


**2.4 训练结果**
|加速卡数量| 模型  |  混合精度 | Batch_Size  |  Shape |  AccTop1 |
|---|---|---|---|---|---|
|  1 | ACNet  |  是 |  256 |  224*224  |  42.8 |
