# TinyNet

## 1. 模型概述
TinyNet 是一种轻量级神经网络架构，来自论文《Model Rubik's Cube: Twisting Resolution, Depth and Width for TinyNets》，由 Gao Huang 等人在 2020 年发表。TinyNet 通过联合优化网络的输入分辨率（resolution）、深度（depth）和宽度（width），在低计算复杂度下实现高效的图像分类性能。其设计目标是针对资源受限的设备（如移动设备、物联网设备），在保持合理精度的同时显著降低计算量（FLOPs）和参数量。TinyNet 基于 EfficientNet 的复合缩放思想，提出了一种专门针对小型网络的系统性缩放策略，生成了一系列高效模型（如 TinyNet-A、B、C 等）。

本项目适配了 TinyNet 模型，提供在 PyTorch 框架下的训练和微调支持，适用于 ImageNet 数据集下的分类任务等场景

## 2. 快速开始
使用 TinyNet 模型执行训练的主要流程如下：
1. 基础环境安装：完成训练前的环境检查和安装。
2. 获取数据集：获取训练所需的数据集。
3. 构建环境：配置模型运行环境。
4. 启动训练：运行训练脚本。

### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
TinyNet 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https：//image-net.org/) 下载。


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
cd <ModelZoo_path>/PyTorch/contrib/Classification/tinynet/run_scripts
```
2. 运行训练. 该模型支持单机单卡。
```shell
python run_tinynet.py --data_path /data/teco-data/imagenet --batch_size 64 --epochs 1 --lr 0.1 --save_path ./checkpoints --num_steps 100
```
更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 
![训练loss曲线](./run_scripts/loss.jpg)

MeanRelativeError: ---需要修改---
MeanAbsoluteError: ---需要修改---
---需要修改--- mean_relative_error=np.float64(0.0005817275664799443) <= 0.05 or mean_absolute_error=np.float64(0.003737578392028809) <= 0.0002