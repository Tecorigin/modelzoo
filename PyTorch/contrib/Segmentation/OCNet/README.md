# OCNet
## 1. 项目介绍
OCNet（Object Context Network）是一种基于深度学习的语义分割网络，专注于通过引入对象上下文（Object Context）的概念来提升分割精度。OCNet的核心思想是利用自注意力机制（Self-Attention Mechanism），通过学习像素之间的相似性来近似对象上下文，从而增强特征表示.


## 2. 快速开始

### 2.1 基础环境安装
1. 安装python依赖
``` bash
cd OCNet
# 克隆基础torch环境
conda create -n SegmenTron --clone torch_env
# install requirements
pip install -r requirements.txt
```
2. 下载权重文件[resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)并放到`OCNet/checkpoints`路径下

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
    cd OCNet/scripts
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
|1| OCNet|是|4|768*768| / |

