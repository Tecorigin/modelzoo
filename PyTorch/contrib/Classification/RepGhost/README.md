# RepGhost

## 1. 模型概述
RepGhost 是一种高效的轻量级卷积神经网络架构，来自论文《RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization》，由程鹏晨等人于2022年发表。RepGhost 通过结构重参数化技术优化了 GhostNet 的特征复用机制，旨在在资源受限的移动设备上实现高效的图像分类，同时降低推理延迟。本项目适配了 RepGhost 模型，提供在 PyTorch 框架下的训练，适用于 ImageNet 数据集下的分类任务等场景。

## 2. 快速开始
使用 RepGhost 模型执行训练的主要流程如下：
1. 基础环境安装：完成训练前的环境检查和安装。
2. 获取数据集：获取训练所需的数据集。
3. 构建环境：配置模型运行环境。
4. 启动训练：运行训练脚本。

### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
RepGhost 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https：//image-net.org/) 下载。


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
cd <ModelZoo_path>/PyTorch/contrib/Classification/RepGhost/run_scripts
```
2. 运行训练. 该模型支持单机单卡。
```shell
python3 run_repghost.py --data_path /data/teco-data/imagenet --batch_size 64 --epochs 1 --lr 0.1 --save_path ./checkpoints --num_steps 100
```
更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 
![训练loss曲线](./run_scripts/loss.jpg)

MeanRelativeError: -0.0005399692991807783
MeanAbsoluteError: -0.0038007307052612305
pass mean_relative_error=-0.0005399692991807783 <= 0.05 or mean_absolute_error=-0.0038007307052612305 <= 0.0002