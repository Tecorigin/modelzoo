# ENCNet
## 1. 项目介绍

ERFNet（Efficient Residual Factorized Network）是一种高效的语义分割网络，专为实时应用设计。它通过引入高效的残差分解卷积（Residual Factorized Convolution）和轻量级的编码器-解码器结构，显著降低了计算复杂度，同时保持了较高的分割精度。ERFNet的设计目标是实现实时语义分割，适用于资源受限的环境，如移动设备和嵌入式系统。该网络在多个语义分割任务中表现出色，尤其是在需要快速处理的场景中，如城市景观图像分割和自动驾驶。


## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd ERFNet
# 克隆基础torch环境
conda create -n erfnet --clone torch_env
# install requirements
pip install -r requirements.txt
```

### 2.2 数据集准备

建议将数据集根目录设置为`$data/datasets`.
```
data
|-- datasets
|   |-- cityscapes
|   |   |-- gtFine
|   |   |   |-- test
|   |   |   |-- train
|   |   |   `-- val
|   |   `-- leftImg8bit
|   |       |-- test
|   |       |-- train
|   |       `-- val

```
可以到 [Cityscape](https://www.cityscapes-dataset.com) 注册账户下载数据集.

### 2.3 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd ERFNet/train
    ```

- 单机单SPA训练
    ```
    python train.py
    ```
- 单机单卡训练（DDP）
    ```
    torchrun --nproc_per_node 4 train.py
    ```


### 2.5 训练结果


|加速卡数量  |模型 | 混合精度 |Batch size|Shape| AccTop1|
|:-:|:-:|:-:|:-:|:-:|:-:|
|1| ERFNet|是|6|768*768| / |