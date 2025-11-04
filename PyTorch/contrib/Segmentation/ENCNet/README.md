# ENCNet
## 1. 项目介绍

ENCNet（Enhanced Contextual Network）是一种专注于上下文建模的语义分割网络，通过引入增强的上下文模块（Enhanced Context Module, ECM）来捕捉长距离依赖关系，从而提升分割精度。ENCNet的设计目标是通过高效的上下文建模和特征融合，实现在复杂场景下的高精度语义分割，同时保持合理的计算复杂度。该网络在多个语义分割任务中表现出色，尤其是在需要精细分割的场景中，如城市景观图像分割和医学图像分割。


## 2. 快速开始

### 2.1 基础环境安装
1. 安装python依赖
``` bash
cd ENCNet
# 克隆基础torch环境
conda create -n SegmenTron --clone torch_env
# install requirements
pip install -r requirements.txt
```
2. 下载权重文件[resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)并放到`ENCNet/checkpoints`路径下

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
    cd ENCNet/scripts
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
|1| ENCNet|是|4|768*768| / |