# EfficientNet

## 1. 模型概述
EfficientNet 是由谷歌的研究团队在 2019 年提出的一种新型卷积神经网络（CNN）架构，旨在优化模型的性能和效率。其主要目标是通过减少计算成本和参数数量，同时提高模型的准确性和可扩展性。

源码链接: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test9_efficientNet

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

#### 2.2.1 数据集位置
/mnt/nvme/common/train_dataset/mini-imagenet

#### 2.2.2 数据集目录结构

mini_imagenet数据集目录结构参考如下所示:

```
|-mini-imagenet
    |-images
        |-n1313361300001299.jpg
        |-n1313361300001297.jpg
        |-...
    |-train_list.txt
    |-val_list.txt
```

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/EfficientNet

conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"
```
<p align="center">
    <img src="images/env.png" alt="Source Image" width="80%">
</p>

2. 安装python依赖
``` 
bash
# install requirements
pip install -r requirements.txt
```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/EfficientNet
    ```
训练过程保存的best权重会保存在"weights"中,日志会保存在工作目录中。

- 单机单SPA训练
    ```
    python run_scripts/run_efficientnet.py --batch_size 64 --epochs 100 --distributed False --num_classes 100 --lr 0.1 --autocast True --dataset_path /mnt/nvme/common/train_dataset/mini-imagenet
    ```
- 单机单卡训练（DDP）
    ```
    python run_scripts/run_efficientnet.py --nproc_per_node 4 --batch_size 64 --epochs 100 --distributed True --num_classes 100  --dataset_path /mnt/nvme/common/train_dataset/mini-imagenet --lr 0.1 --autocast True
    ```

更多训练参数参考[README](run_scripts/README.md)


### 2.5 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| EfficientNet |是|64|300*300|

**训练结果量化指标如下表所示**

| 训练数据集 | 输入图片大小 |nproc_per_node|accuracy(DDP)|
| :-----: | :-----: |:------: |:------: |
| mini_imagenet | 224x224 |4|79.6% |

**训练过程loss曲线如下图所示**
<p align="center">
    <img src="images/loss_curve.png" alt="Source Image" width="55%">
</p>

**训练过程正确率曲线如下图所示**
<p align="center">
    <img src="images/accuracy_curve.png" alt="Source Image" width="55%">
</p>


