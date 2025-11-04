# FPENet
## 1. 项目介绍
FPENet是一种高效的轻量级语义分割网络，它通过特征金字塔编码模块（FPE）和互嵌上采样模块（MEU）实现了多尺度特征的高效提取和融合，显著提升了分割精度和推理速度.FPENet的设计目标是优化计算复杂度和内存占用，使其能够在资源受限的环境中实现实时语义分割.该网络在Cityscapes和CamVid等数据集上表现出色，尤其在高分辨率图像分割任务中展现了强大的性能.


## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd FPENet
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
    cd FPENet/scripts
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
|1| FPENet|是|4|768*768| / |

