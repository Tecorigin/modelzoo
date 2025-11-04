# MobileSeg

## 1. 模型概述

MobileSeg是一种专为移动设备和边缘计算场景设计的轻量级语义分割模型，通过创新的编解码架构和高效特征融合技术，在保证分割精度的同时显著降低计算开销。其核心设计包含多级StrideFormer骨干网络，结合动态跨步注意力机制，能够以最小的参数量提取多尺度语义与细节特征；聚合注意力模块（AAM）通过语义特征集成投票策略增强上下文信息融合，而有效插值模块（VIM）仅对预测存在的类别进行上采样，大幅减少模型延迟。该模型在ADE20K、Cityscapes等数据集上展现出优异的速度-精度平衡，适用于自动驾驶实时场景分割、机器人环境感知、遥感图像道路提取等对轻量化和低功耗要求较高的应用领域，凭借其高效的推理能力和灵活的部署特性，成为移动端视觉任务的重要解决方案。

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
cd PaddlePaddle/contrib/Segmentation/MobileSeg
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
    cd PaddlePaddle/contrib/Segmentation/MobileSeg
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    export PADDLE_XCCL_BACKEND=sdaa
    python -m  paddle.distributed.launch --devices=0,1,2,3 tools/train.py  \
        --config configs/mobileseg/mobileseg_mobilenetv2_cityscapes_1024x512_80k.yml \
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
| 4 | MobileSeg | 是 | 4 | 1620 | 2.7931 |