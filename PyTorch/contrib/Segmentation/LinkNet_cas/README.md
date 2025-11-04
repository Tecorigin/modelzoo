# LinkNet

## 1. 模型概述
LinkNet是一种轻量级的语义分割网络，它通过采用编码-解码结构，利用跳跃连接来合并编码器和解码器中的特征，从而减少信息丢失，增强特征传递。LinkNet的设计目标是实现高效的语义分割，同时保持较低的计算复杂度，适用于资源有限的环境。该方法在多个语义分割任务上表现出色，尤其是在城市景观图像分割中。

## 2. 快速开始

### 2.1 基础环境安装
1. 安装python依赖
``` bash
cd LinkNet_cas
# 克隆基础torch环境
conda create -n linknet --clone torch_env
# install requirements
pip install -r requirements.txt
```
2. 下载权重文件[resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth)并放到`LinkNet_cas/checkpoints`路径下

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
    cd LinkNet_cas
    ```
- 单机单SPA训练
    ```
    python main_plot.py --datapath /data/datasets/cityscapes/
    ```
- 单机单卡训练（DDP）
    ```
    torchrun --nproc_per_node 4 main_plot.py --datapath /data/datasets/cityscapes/
    ```


### 2.5 训练结果

|加速卡数量  |模型 | 混合精度 |Batch size|Shape| AccTop1|
|:-:|:-:|:-:|:-:|:-:|:-:|
|1| LinkNet|是|8|512*512| / |


