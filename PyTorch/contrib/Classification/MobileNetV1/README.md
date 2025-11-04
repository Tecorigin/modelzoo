
# MobilenetV1

## 1.模型概述
MobileNetV1 是一种轻量级卷积神经网络模型，专为移动设备和嵌入式设备设计。它通过使用深度可分离卷积（Depthwise Separable Convolution）替代传统卷积操作，大幅减少了模型的计算量和参数量。深度可分离卷积将标准卷积分解为深度卷积和逐点卷积两个步骤，既保留了特征提取能力，又显著降低了计算复杂度。

MobileNetV1 的亮点在于其高效的设计，能够在保证较高精度的同时，显著提升模型的运行速度。它适用于计算资源受限的场景，如移动端图像分类、目标检测等任务。尽管模型结构简单，但 MobileNetV1 在多个基准数据集上表现优异，为轻量级神经网络的设计提供了重要参考。MobileNetV1 的出现推动了深度学习在移动端的应用，为实时计算和低功耗场景提供了高效的解决方案。其设计思想也影响了后续轻量级网络的发展，成为移动端深度学习模型的经典代表之一。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

MobilenetV1运行在ImageNet数据集上，数据集配置可以参考
https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 启动训练

准备环境
```
cd <Modelzoo_path>/PyTorch/contrib/Classification/MobileNetV1
pip install -r requirements.txt
```

模型单机单卡训练：
```
cd <Modelzoo_path>/PyTorch/contrib/Classification/MobileNetV1

./distributed_train.sh 4 --data-dir /data/datasets/ --output /data/ckpt/ \
    --model mobilenetv1_100 --sched cosine --epochs 1 --warmup-epochs 0 \
    --lr 0.4 --reprob 0.5 --remode pixel --batch-size 32 --amp -j 4 \
    2>&1 | tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练约5~6小时左右后，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | MobilenetV1 | Amp | 32 | Epoch：1 | 5.62 | 22.222 |