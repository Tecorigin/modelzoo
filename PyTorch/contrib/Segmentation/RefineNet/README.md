# RefineNet
## 1. 项目介绍
RefineNet基于编码器-解码器架构，编码器部分通常采用预训练的ResNet网络，分为多个卷积块。与之对应的是级联的RefineNet单元，每个单元都通过长程残差连接（long-range residual connections）将编码器的特征图与解码器的特征图进行融合。这种设计确保了从低级到高级的特征信息能够被充分利用，同时通过短程残差连接（short-range residual connections）增强了网络的训练能力.


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
2. 下载权重文件[resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)并放到`RefineNet/checkpoints`路径下

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
    cd RefineNet/scripts
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
|1| RefineNet|是|4|769*769| / |

