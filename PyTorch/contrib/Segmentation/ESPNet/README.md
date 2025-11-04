# ESPNet
## 1. 项目介绍

ESPNet是一种轻量级的语义分割网络，专为高效计算和实时应用设计。它通过独特的编码器-解码器结构和高效的特征提取模块，实现了低计算复杂度和高分割精度.


## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd ESPNet
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
    cd ESPNet/scripts
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
|1| ESPNet|是|4|768*768| / |

