# DFANet
## 1. 项目介绍
DFANet（Deep Feature Aggregation Network）是一种高效的语义分割网络，专为实现实时语义分割而设计。它通过深度特征聚合模块（Deep Feature Aggregation Module）和轻量级的编码器-解码器结构，实现了高效的信息传递和特征融合。DFANet的设计目标是减少计算复杂度，同时保持较高的分割精度，适用于资源受限的环境，如移动设备和嵌入式系统。该网络在城市景观图像分割等任务中表现出色，能够快速准确地分割出道路、建筑物等关键元素。


## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd DFANetNet
# 克隆基础torch环境
conda create -n SegmenTron --clone torch_env
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
    cd DFANet/scripts
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
|1| DFANet|是|2|768*768| / |

