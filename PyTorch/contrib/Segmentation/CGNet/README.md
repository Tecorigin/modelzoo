# CGNet
## 1. 项目介绍

CGNet是一种专为语义分割设计的轻量级网络，通过融合局部特征、周围上下文和全局上下文信息来提升分割精度。其核心是CG块（Context Guided Block），该模块包含局部特征提取器、周围上下文特征提取器、联合特征提取器和全局特征提取器。CGNet的网络架构分为三个阶段：第一个阶段使用卷积层提取特征，第二和第三阶段堆叠多个CG块，并通过输入注入机制进一步加强特征传递。CGNet的设计目标是减少参数数量和内存占用，同时提高分割精度，特别适合移动设备等资源受限的场景。

## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd CGNet
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
    cd CGNet/scripts
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
|1| CGNet|是|8|768*768| / |

