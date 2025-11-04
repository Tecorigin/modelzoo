# RepVGG

## 1.模型概述
RepVGG是一种深度卷积神经网络模型，它在VGG网络的Block块中加入了Identity和残差分支，相当于把ResNet网络中的精华应用 到VGG网络中。在模型推理阶段，它通过Op融合策略将所有的网络层都转换为Conv3*3，便于模型的部署与加速。RepVGG的亮点在于网络训练和网络推理阶段使用不同的网络架构，使得训练阶段更关注精度，推理阶段更关注速度。

RepVGG模型在各种计算机视觉任务中表现优异，在图像分类、目标检测等任务上均取得了当时最先进的结果。独特的架构设计使得 RepVGG 在训练时可以享受多分支结构带来的性能优势，同时在推理时又可以保持单路结构的效率。因而 RepVGG 是一种兼顾性能和效率的优秀卷积神经网络模型，它的出现为深度学习在实际应用场景中的部署提供了新的思路和选择。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

RepVGG运行在ImageNet数据集上，数据集配置可以参考
https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 启动训练

模型单机单机单卡训练：
```
python -m torch.distributed.launch --nproc_per_node 4 main.py --arch RepVGGplus-L2pse --batch-size 32 --tag my_experiment --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.005 TRAIN.WEIGHT_DECAY 0.1 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.MOMENTUM 0.9 TRAIN.WARMUP_LR 5e-7 TRAIN.MIN_LR 0.0 TRAIN.WARMUP_EPOCHS 0 AUG.PRESET raug15 AUG.MIXUP 0.4 AUG.CUTMIX 1.0 DATA.IMG_SIZE 256 2>&1|tee train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练总耗时为2小时，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | RepVGG | Amp | 32 | Iterations：800/10009 | 8.9866 | / |