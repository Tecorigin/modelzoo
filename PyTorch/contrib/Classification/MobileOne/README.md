
# MobileOne

## 1.模型概述
MobileOne是一种高效的卷积神经网络模型，专为移动设备和边缘计算场景设计。它通过结构重参数化技术，在训练阶段引入多分支结构以提升模型性能，而在推理阶段则将这些分支合并为单一卷积层，从而实现高效的推理速度。MobileOne的核心思想是在保证精度的同时，显著降低计算复杂度和内存占用。

MobileOne在图像分类、目标检测等任务中表现出色，尤其在资源受限的设备上展现了优越的性能。其独特的架构设计使得模型在训练时能够充分利用多分支的优势，而在推理时则通过结构简化实现高效部署。因此，MobileOne是一种兼顾精度与效率的轻量级网络模型，为移动端和边缘计算场景中的深度学习应用提供了新的解决方案。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

MobileOne运行在ImageNet数据集上，数据集配置可以参考
https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 启动训练

准备环境：

```
cd <Modelzoo_path>/PyTorch/contrib/Classification/MobileOne

pip install -r requirements.txt
```

模型单机单卡训练：
```
cd <Modelzoo_path>/PyTorch/contrib/Classification/MobileOne

./distributed_train.sh 4 --data-dir /data/datasets/ --output /data/ckpt/ \
    --model mobileone_s0 --sched cosine --epochs 1 --warmup-epochs 0 \
    --lr 0.4 --reprob 0.5 --remode pixel --batch-size 32 --amp -j 4 \
    2>&1 | tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练约5~6小时左右后，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | MobileOne | Amp | 32 | Epoch：1 | 6.35 | 6.144 |