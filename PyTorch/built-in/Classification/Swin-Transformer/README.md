###  Swin-Transformer

**1.模型概述** 

Swin Transformer（名称Swin代表S移位窗口）最初在https://arxiv.org/abs/2103.14030中描述，它可以作为计算机视觉的通用骨干。它基本上是一个分层的 Transformer，其表示是用移位窗口计算的。移位窗口方案通过将自注意力计算限制在非重叠的本地窗口上，同时还允许跨窗口连接，从而提高效率。

**2.快速开始**

使用本模型执行训练的主要流程如下：

基础环境安装：介绍训练前需要完成的基础环境检查和安装。

获取数据集：介绍如何获取训练所需的数据集。

启动训练：介绍如何运行训练。

**2.1 基础环境安装**

注意激活自身环境
（注意克隆torch.sdaa库）

**2.2 获取数据集**

imagenet数据集可以在官网进行下载；共享存储路径：/mnt/dataset/imagenet


**2.3 启动训练**

1.安装依赖

    pip install timm==0.4.12
    pip install opencv-python termcolor==1.1.0 yacs==0.1.8 pyyaml scipy

2.运行指令

**单机单卡**
  cd kernels/window_process
  python setup.py install
  export TORCH_SDAA_AUTOLOAD=cuda_migrate
  python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path /mnt/nvme1/dataset/datasets/resnet50 --batch-size 64  
    
