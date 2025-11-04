
# MobilenetV2

## 1.模型概述
MobileNetV2 是一种轻量级卷积神经网络模型，专为移动设备和嵌入式设备设计。它在 MobileNetV1 的基础上引入了倒残差结构（Inverted Residuals）和线性瓶颈层（Linear Bottleneck），显著提升了模型的效率和性能。倒残差结构通过先扩展通道数再压缩的方式，增强了特征表达能力，而线性瓶颈层则避免了非线性激活函数对低维特征的破坏。

MobileNetV2 在保持较低计算复杂度的同时，能够实现较高的精度，适用于图像分类、目标检测和语义分割等任务。其轻量化的设计使得模型在资源受限的设备上也能高效运行，为移动端和边缘计算场景提供了优秀的解决方案。MobileNetV2 的出现进一步推动了深度学习在实际应用中的普及和落地。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

MobilenetV2运行在ImageNet数据集上，数据集配置可以参考
https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 启动训练

准备环境:
```
cd <Modelzoo_path>/PyTorch/contrib/Classification/MobileNetV2

pip install -r requiments.txt
```

模型单机单卡训练：
```
cd <Modelzoo_path>/PyTorch/contrib/Classification/MobileNetV2

./distributed_train.sh 4 --data-dir /data/datasets/ --output /data/ckpt/ \
    --model mobilenetv2_100 --sched cosine --epochs 1 --warmup-epochs 0 \
    --lr 0.4 --reprob 0.5 --remode pixel --batch-size 32 --amp -j 4 \
    2>&1 | tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练约5~6小时左右后，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | MobilenetV2 | Amp | 32 | Iterations：6000 | 6.05 | / |