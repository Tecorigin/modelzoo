# LedNet
## 1. 项目介绍

LEDNet是一种轻量级的语义分割网络，采用不对称的编码器-解码器架构，专注于实现实时语义分割。其编码器部分基于ResNet作为骨干网络，并引入了通道分割和通道洗牌操作，显著降低了计算成本，同时保持了较高的分割精度。解码器部分则采用了注意力金字塔网络（APN），进一步减轻了网络的复杂性。

## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd LedNet
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
    cd LedNet/scripts
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
|1| LedNet|是|8|768*768| / |

