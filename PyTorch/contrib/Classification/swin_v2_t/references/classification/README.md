###  Swin_v2_t

**1.模型概述** 

Swin Transformer V2 是一种先进的视觉模型，由 Ze Liu 等人提出，旨在通过扩展容量和分辨率来提升计算机视觉任务的表现。Swin Transformer V2 在多个方面对原始的 Swin Transformer 进行了改进，包括残差后归一化方法结合余弦注意力，以提高训练稳定性；对数间隔的连续位置偏差方法，以更好地将低分辨率预训练模型迁移到高分辨率任务；以及自监督预训练方法 SimMIM，以减少对大量标记数据的需求。
Swin Transformer V2 提供了多种模型变体，其中 SwinV2-Tiny 是较小的版本，适用于快速实验和轻量级任务。该模型在图像分类、目标检测等多个视觉任务上表现出色，且在训练效率和数据需求方面相比其他大规模视觉模型具有显著优势。

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

注意此时使用四卡，且指定了模型名称为Swin_v2_t以及数据集的地址（数据集里面需要包括train和val文件）

**单机单核组**(cd 到指定路径下)

    cd ../references/classification/
    python  train.py --model swin_v2_t  --world-size 1 --epochs 300 --batch-size 64 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 256 --val-crop-size 256 --train-crop-size 256 

**单机单卡**

    torchrun --nproc_per_node=4 train.py --model swin_v2_t --epochs 300 --batch-size 64 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 256 --val-crop-size 256 --train-crop-size 256 

**2.4 训练结果**
|加速卡数量| 模型  |  混合精度 | Batch_Size  |  Shape |  AccTop1 |
|---|---|---|---|---|---|
|  1 | Swin_v2_t  |  是 |  64 |  224*224  |  0.268 |
