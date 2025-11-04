# DenseASPP
## 1. 项目介绍
DenseASPP（Dense Atrous Spatial Pyramid Pooling）是一种用于语义分割的网络模块，通过密集连接的空洞卷积层生成覆盖更大尺度范围且更密集的多尺度特征，显著提升了分割精度。它结合了平行和级联的空洞卷积优点，解决了传统ASPP在大扩张率下特征退化的问题。DenseASPP通过密集连接的方式，使得每一层的输出都参与到后续层的计算中，增强了特征的传播和利用，同时通过1×1卷积控制模型大小和计算量。


## 2. 快速开始

### 2.1 基础环境安装
1. 安装python依赖
``` bash
cd RefineNet
# 克隆基础torch环境
conda create -n SegmenTron --clone torch_env
# install requirements
pip install -r requirements.txt
```
2. 下载权重文件[resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)并放到`DenseASPP_cas/checkpoints`路径下

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
    cd DenseASPP_cas/scripts
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
|1| DenseASPP|是|4|768*768| / |

