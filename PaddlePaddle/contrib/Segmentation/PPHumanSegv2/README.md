# PP-HumanSeg v2

## 1. 模型概述

PPHumanSeg 是一款专门用于人体分割的模型，基于飞桨深度学习框架开发。它采用了先进的神经网络架构，能够快速且准确地将人体从复杂背景中分割出来。该模型在大量人体图像数据上进行训练，具备很强的泛化能力，无论是在不同的拍摄场景、光照条件还是人体姿态下，都能有出色的表现。PPHumanSeg 还支持多种应用场景，如人像抠图、视频直播中的背景替换、健身动作分析等，为相关领域提供了高效、便捷的人体分割解决方案，通过简单的 API 调用，开发者就能轻松实现人体分割功能，推动了人体相关图像处理应用的发展。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 mini_supervisely 的数据集。根据[官方文档](./contrib/PP-HumanSeg/README_cn.md)准备数据集。


#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:

```
mini_supervisely/
|-- Annotations
|-- Images
|-- test.txt
|-- train.txt
`-- val.txt
```

### 2.3 构建环境

参考[官方安装说明](./docs/install_cn.md)进行安装

1. 执行以下命令，启动虚拟环境。
``` bash
cd PaddlePaddle/contrib/Segmentation/PPHumanSegv2
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
    cd PaddlePaddle/contrib/Segmentation/PPHumanSegv2
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    export PADDLE_XCCL_BACKEND=sdaa
    python -m  paddle.distributed.launch --devices=0 tools/train.py  \
        --config contrib/PP-HumanSeg/configs/human_pp_humansegv2_lite.yml \
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
| 加速卡数量 | 模型 | 混合精度 | Batch Size | epoch | train_loss |
| --- | --- | --- | --- | --- | --- |
| 1 | PPHumanSegv2 | 是 | 8 | 40 | 0.3427 |