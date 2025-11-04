# FastFCN
## 1. 项目介绍
FastFCN是一种高效且灵活的语义分割框架，通过重新思考骨干网络中的扩张卷积，实现了显著的性能提升和加速效果。其核心创新在于联合金字塔上采样（Joint Pyramid Upsampling, JPU）模块，该模块通过融合多尺度特征，有效提升了分割精度和速度。FastFCN适用于自动驾驶、医学图像分析等需要高精度语义分割的场景。


## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd FastFCN
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
    cd FastFCN/scripts
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
|1| FastFCN|是|12|512*1024| / |

