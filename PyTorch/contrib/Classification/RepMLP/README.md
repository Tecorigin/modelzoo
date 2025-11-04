## RepMLP

## 1.模型概述
RepMLP是一种用于图片识别的多层感知器风格的神经网络，它为了结合MLP在长依赖以及位置感知方面比较奏效和卷积Conv善于局部感知的特点，在训练阶段结合MLP和Conv网络单元，使网络同时具备全局感知和局部感知能力；在推理阶段利用重参化技术将Conv合并到MLP中去以提高模型推理效率。

RepMLP模型在各种计算机视觉任务中表现优异，在图像分类、人脸识别以及语义分割等任务(无论是否具有平移不变性)上均能涨点。它结合FC层的全局表征能力和位置先验性质以及卷积层的局部先验性质，可以在大幅增加参数的同时不会造成推理速度的显著降低。因而 RepMLP 是一种兼顾性能和效率的优秀卷积神经网络模型，已成为深度学习和计算机视觉领域的重要基础模型之一。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

RepMLP运行在ImageNet数据集上，数据集配置可以参考
https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 启动训练

模型单机单卡训练：
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 65501 main_repmlp_sdaa.py --arch RepMLPNet-T256 --batch-size 16 --tag my_experiment --opts TRAIN.EPOCHS 1 TRAIN.BASE_LR 0.005 TRAIN.WEIGHT_DECAY 0.1 TRAIN.OPTIMIZER.NAME adamw TRAIN.OPTIMIZER.MOMENTUM 0.9 TRAIN.WARMUP_LR 5e-7 TRAIN.MIN_LR 0.0 TRAIN.WARMUP_EPOCHS 0 AUG.PRESET raug15 AUG.MIXUP 0.4 AUG.CUTMIX 1.0 DATA.IMG_SIZE 256 2>&1|tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练总耗时为2小时，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | RepMLP | Amp | 16 | Iterations：1930/20018 | 6.8978 | / |