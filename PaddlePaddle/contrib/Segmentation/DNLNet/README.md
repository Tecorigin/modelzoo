# DNLNet

## 1. 模型概述

DNLNet即Dynamic Normalization and Localization Network，是一种专为视觉任务设计的神经网络架构，主要用于目标检测、语义分割等场景。它创新性地结合了动态归一化（Dynamic Normalization）和定位（Localization）机制，通过自适应调整特征归一化参数和增强空间定位能力，提升模型对复杂场景的理解精度。该网络在骨干网络基础上引入动态归一化层，根据输入特征动态计算归一化参数，有效增强特征的判别性；同时通过定位模块优化目标位置信息的提取，减少背景干扰。在实际应用中，DNLNet尤其擅长处理小目标检测、密集场景分割等任务，例如遥感图像中的车辆识别、医学影像中的病灶定位等。凭借其在动态特征处理和精准定位方面的优势，DNLNet为提升视觉任务的性能提供了新的解决方案，成为计算机视觉领域中优化特征表达与空间建模的重要模型之一。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 cityscapes 数据集。根据[官方文档](./pre_data_cn.md)准备数据集。


#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:

```
cityscapes/
|-- gtFine
|   |-- test
|   |-- train
|   `-- val
`-- leftImg8bit
    |-- test
    |-- train
    `-- val
```

### 2.3 构建环境

参考[官方安装说明](./docs/install_cn.md)进行安装

1. 执行以下命令，启动虚拟环境。
``` bash
cd PaddlePaddle/contrib/Segmentation/DNLNet
conda activate paddleseg
```
2. 安装依赖
``` bash
cd PaddleSeg
pip install -r requirements.txt
pip install -v -e .
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PaddlePaddle/contrib/Segmentation/DNLNet
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    export PADDLE_XCCL_BACKEND=sdaa
    export CUSTOM_DEVICE_BLACK_LIST=conv2d
    python -m  paddle.distributed.launch --devices=0,1,2,3 tools/train.py  \
        --config ./configs/dnlnet/dnlnet_resnet50_os8_cityscapes_1024x512_80k.yml \
        --save_interval 50 \
        --save_dir output 
    ```


### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
| 加速卡数量 | 模型 | 混合精度 | Batch Size | iter | train_loss |
| --- | --- | --- | --- | --- | --- |
| 4 | DNLNet | 是 | 2 | 370 | 0.7362 |