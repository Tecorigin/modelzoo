# DANet
## 1. 项目介绍
DANet（Dual Attention Network）是一种用于语义分割的高效网络架构，它通过引入双重注意力机制，显著提升了特征提取和分割精度。DANet的核心在于其独特的双重注意力模块，能够同时捕捉通道（Channel）和空间（Spatial）维度上的重要信息，从而更精准地进行语义分割。


## 2. 快速开始

### 2.1 基础环境安装
1. 安装python依赖
``` bash
cd DANet
# 克隆基础torch环境
conda create -n SegmenTron --clone torch_env
# install requirements
pip install -r requirements.txt
```
2. 下载权重文件[resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)并放到`DANet/checkpoints`路径下

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
    cd DANet/scripts
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
|1| DANet|是|2|768*768| / |

