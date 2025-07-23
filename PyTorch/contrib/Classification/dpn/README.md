# DPN

## 1. 模型概述
DPN（Dual Path Network）是一种高效的卷积神经网络架构，来自论文 Dual Path Networks，由陈云鹏（Yunpeng Chen）等人在 2017 年发表。DPN 通过结合 ResNet 的残差连接（Residual Connections）和 DenseNet 的密集连接（Dense Connections），设计了双路径结构（Dual Path），在特征复用和特征探索之间取得平衡，从而显著提升了图像分类、目标检测和语义分割等任务的性能，同时保持较高的计算效率。原始代码仓库位于 https://github.com/cypw/DPNs。

本项目适配了 DPN 模型，提供在 PyTorch 框架下的训练和微调支持，适用于imagenet数据集下的分类任务等场景。

## 2. 快速开始
使用 DPN 模型执行训练的主要流程如下：
1. 基础环境安装：完成训练前的环境检查和安装。
2. 获取数据集：获取训练所需的数据集。
3. 构建环境：配置模型运行环境。
4. 启动训练：运行训练脚本。

### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
DPN 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https：//image-net.org/) 下载。


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
cd <ModelZoo_path>/PyTorch/contrib/Classification/dpn/run_scripts
```
2. 运行训练. 该模型支持单机单卡。
```shell
python run_dpn.py --batch_size 128 --num_workers 4 --lr 0.05 --epoch 1 --num_steps 100 --device_num 1
```
更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 
![训练loss曲线](./run_scripts/loss.jpg)

MeanRelativeError: -0.0019443976784576553
MeanAbsoluteError: -0.013461603976712368
pass mean_relative_error=np.float64(-0.0019443976784576553) <= 0.05 or mean_absolute_error=np.float64(-0.013461603976712368) <= 0.0002