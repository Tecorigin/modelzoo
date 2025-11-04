# Unet++

## 1. 模型概述

Unet++是一种专为医学图像分割等高精度任务设计的改进型语义分割网络，通过引入嵌套的密集跳接连接（Nested Dense Skip Connections）和级联损失函数（Cascaded Loss），有效解决了传统Unet特征融合不足的问题。其网络结构在Unet基础上构建了多尺度密集块，允许浅层细节特征与深层语义特征在多个分辨率层次上进行密集交互，显著提升了模型对边界细节和复杂结构的分割能力。同时，级联损失机制通过监督中间层预测结果，实现了模型的渐进式训练，进一步优化了特征学习过程。Unet++在医学影像领域表现尤为突出，例如在肿瘤区域分割、细胞检测等任务中展现出优异性能，同时也适用于遥感图像解译、工业缺陷检测等需要精细分割的场景。凭借其在特征复用和层次化特征融合方面的创新，Unet++成为语义分割领域兼具精度与灵活性的重要模型之一，为复杂场景下的精准分割提供了有力工具。

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
cd PaddlePaddle/contrib/Segmentation/UnetPP
conda activate paddleseg
```
2. 安装依赖
``` bash
pip install -r requirements.txt
pip install -v -e .
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PaddlePaddle/contrib/Segmentation/UnetPP
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    export PADDLE_XCCL_BACKEND=sdaa
    python -m  paddle.distributed.launch --devices=0,1,2,3 tools/train.py  \
        --config configs/unet_plusplus/unet_plusplus_cityscapes_1024x512_160k.yml \
        --save_interval 50 \
        --save_dir output \
        --precision fp16 
    ```


### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
| 加速卡数量 | 模型 | 混合精度 | Batch Size | iter | train_loss |
| --- | --- | --- | --- | --- | --- |
| 2 | unet++ | 是 | 2 | 530 | 1.6398 |