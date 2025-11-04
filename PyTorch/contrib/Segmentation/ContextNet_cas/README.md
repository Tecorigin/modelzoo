# ContextNet
## 1. 项目介绍
ContextNet是一种用于语义分割的轻量级卷积神经网络，旨在通过融合局部和全局上下文信息来提高分割性能。ContextNet采用了一种双路径结构，其中主路径处理高分辨率细节信息，辅助路径则通过下采样和扩展卷积来捕捉全局上下文信息。两条路径的信息最终在解码阶段进行融合，从而在保持计算效率的同时，提高了语义分割的准确性。


## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd ContextNet_cas
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
    cd ContextNet_cas/scripts
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
|1| ContextNet|是|4|768*768| / |

