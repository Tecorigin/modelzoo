# LCNet

## 1. 模型概述
PP-LCNet 是一种高效的轻量级卷积神经网络架构，来自论文《PP-LCNet: A Lightweight CPU Convolutional Neural Network》，由程翠等人在2021年发表。PP-LCNet基于MKLDNN加速策略，旨在在资源受限的CPU设备（如Intel CPU）上实现高精度、低延迟的图像分类，同时适用于多种计算机视觉任务。本项目适配了 LCNet 模型，提供在 PyTorch 框架下的训练支持，适用于 ImageNet数据集下的分类任务等场景。

## 2. 快速开始
使用 LCNet 模型执行训练的主要流程如下：
1. 基础环境安装：完成训练前的环境检查和安装。
2. 获取数据集：获取训练所需的数据集。
3. 构建环境：配置模型运行环境。
4. 启动训练：运行训练脚本。

### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
LCNet 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https：//image-net.org/) 下载。


#### 2.2.2 处理数据集
具体配置方式可参考：https：//blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    ```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录. 
```
cd <ModelZoo_path>/PyTorch/contrib/Classification/LCNet/run_scripts
```
2. 运行训练. 该模型支持单机单卡。
```shell
python3 run_lcnet.py --data_path /data/teco-data/imagenet --batch_size 64 --epochs 1 --lr 0.1 --save_path ./checkpoints --num_steps 100
```
更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 
![训练loss曲线](./run_scripts/loss.jpg)

MeanRelativeError: -0.0007282508593108522
MeanAbsoluteError: -0.005097827911376953
pass mean_relative_error=-0.0007282508593108522 <= 0.05 or mean_absolute_error=-0.005097827911376953 <= 0.0002