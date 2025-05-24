# ResNet-RS
## 1. 模型概述
ResNet-RS（Re-scaled ResNet）是一种基于残差网络（ResNet）的改进模型，旨在提升图像分类任务的性能和效率。
它由Google Brain和UC Berkeley的研究团队在2021年提出
- 论文链接：[Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579)
- 仓库链接：[nachiket273/pytorch_resnet_rs](https://github.com/nachiket273/pytorch_resnet_rs)

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
ResNet-RS 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/xzxg001/article/details/142465729


### 2.3 构建环境
所使用的环境下已经包含 PyTorch 框架虚拟环境。
1. 执行以下命令，启动虚拟环境。 
```
conda activate torch_env
```

2. 安装python依赖
```
pip install -r requirements.txt
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
```
cd <ModelZoo_path>/PyTorch/contrib/Classification/ResNet-RS/run_scripts
```
2. 运行训练. 该模型支持单机单卡。
```shell
python run_resnetrs.py \
--dataset_path /data/teco-data/imagenet \
--batch_size 32 \
--epochs 1 \
--lr 0.01 \
--amp \
--save_path ../checkpoints \
--max_step 100
--device sdaa
```
更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练 loss 曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）：

![loss](./run_scripts/loss.jpg)

```text
MeanRelativeError: -0.0032978454163112556
MeanAbsoluteError: -0.024992923736572265
Rule,mean_absolute_error -0.024992923736572265
pass mean_relative_error=-0.0032978454163112556 <= 0.05 or mean_absolute_error=-0.024992923736572265 <= 0.0002
```
