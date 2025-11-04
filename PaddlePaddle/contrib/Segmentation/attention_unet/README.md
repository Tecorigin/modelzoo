# Attention U-Net

## 1. 模型概述

Attention U-Net是在经典U-Net架构基础上改进而来的用于图像分割任务的神经网络模型。它将注意力机制引入到U-Net中，在编码和解码阶段的跳跃连接部分加入注意力门控结构。这一创新设计使得模型能够自动聚焦于图像中的重要特征区域，抑制无关信息，增强了对目标对象的特征提取能力。在进行图像分割时，Attention U-Net能够更精准地定位和分割出目标物体，尤其在处理具有复杂背景、目标和周围组织对比度不高的图像时表现出色。该模型在医学图像分割领域应用广泛，如肿瘤、器官的分割，同时也可用于自然场景图像分割等其他领域。凭借注意力机制带来的对特征的高效筛选和利用，Attention U-Net提升了分割的准确性和鲁棒性，成为图像分割任务中极具竞争力的模型之一。 

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
cd PaddlePaddle/contrib/Segmentation/attention_unet
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
    cd PaddlePaddle/contrib/Segmentation/attention_unet
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    export PADDLE_XCCL_BACKEND=sdaa
    export CUSTOM_DEVICE_BLACK_LIST=conv2d
    python -m  paddle.distributed.launch --devices=2,3 tools/train.py  \
        --config configs/attention_unet/attention_unet_cityscapes_1024x512_80k.yml \
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
loss下降趋势不明显
| 加速卡数量 | 模型 | 混合精度 | Batch Size | iter | train_loss |
| --- | --- | --- | --- | --- | --- |
| 2 | attention unet | 是 | 2 | 150 |  1.7648 |