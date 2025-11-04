# DABNet
## 1. 项目介绍
DABNet是一种高效轻量级的语义分割网络，专门针对实时应用场景设计。它采用了创新的双重注意力块（Dual Attention Block，DAB）来增强特征提取能力，同时通过优化网络结构，显著降低了计算复杂度和内存占用。


## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd DABNet
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
    cd DABNet/scripts
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
|1| DABNet|是|4|512*1024| / |

