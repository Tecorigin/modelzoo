# FCN
## 1. 项目介绍
FCN（Fully Convolutional Network）是一种经典的语义分割网络，通过全卷积操作将图像像素映射到像素类别，支持任意尺寸的输入图像。FCN通过跳跃连接融合不同层次的特征，结合高层语义信息和低层细节信息，从而提高分割精度.


## 2. 快速开始

### 2.1 基础环境安装
1. 安装python依赖
``` bash
cd FCN_cas
# 克隆基础torch环境
conda create -n SegmenTron --clone torch_env
# install requirements
pip install -r requirements.txt
```
2. 下载权重文件[resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)并放到`FCN_cas/checkpoints`路径下
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
    cd FCN_cas/scripts
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
|1| FCN|是|4|768*768| / |

