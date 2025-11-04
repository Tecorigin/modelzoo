# HardNet
## 1. 项目介绍

HardNet是一种高效的语义分割网络，其核心在于通过优化网络结构来减少内存访问流量，同时保持高计算效率
。它采用了独特的谐波密集连接（Harmonic Dense Connection）设计，主要使用3x3卷积层，减少内存交互，将网络从内存限制转变为计算密集型。这种设计使得HardNet在实时目标检测和高分辨率视频语义分割等任务中表现出色。

## 2. 快速开始

### 2.1 基础环境安装
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd Project_name/script

conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"
```
2. 安装python依赖
``` bash
# 克隆基础torch环境
conda create -n SegmenTron --clone torch_env
# install requirements
pip install -r requirements.txt
```

### 2.2 数据集准备

建议将数据集根目录设置为`$Project_name/datasets`.
```
Project_name
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
    cd Project_name/script
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
|1| HardNetNet|是|16|1024*1024| / |

