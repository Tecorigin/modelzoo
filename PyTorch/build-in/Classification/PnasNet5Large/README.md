# PnasNet5Large

## 1. 模型概述
PNASNet5Large 是 PNASNet（Progressive Neural Architecture Search Network） 系列中的一个经典模型，属于基于 神经网络架构搜索（NAS, Neural Architecture Search） 技术设计的卷积神经网络（CNN）。该模型由谷歌（Google）团队于 2017 年提出，其核心思想是通过 强化学习（Reinforcement Learning） 或 进化算法 自动搜索最优的神经网络结构，以替代传统人工设计网络的繁琐过程。PNASNet 系列包含多个变体，其中 PNASNet5Large 是针对图像分类任务设计的大型模型，在经典数据集（如 ImageNet）上具有优异的性能，曾是自动化架构搜索领域的代表性成果之一。PnasNet5Large的代码主要从[GitHub]迁移和调整 [GitHub](https://github.com/Cadene/pretrained-models.pytorch).

- 仓库链接：[pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

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
PnasNet5Large运行在ImageNet 1k上，这是一个来自ILSVRC挑战赛的广受欢迎的图像分类数据集。您可以点击[此链接](https://image-net.org/download-images)从公开网站中下载数据集。

#### 2.2.2 处理数据集
· 执行以下命令，解压训练数据集。
```
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

```
· 执行以下命令，解压测试数据并将图像移动到子文件夹中。
```
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

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

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/build-in/Classification/PNASNet5Large
    ```

2. 运行训练。该模型支持单机单卡。

    ```
    #单核组
    bash train_performance.sh --data_path=<imagenet_path> --rank_size=1
    #单卡
    bash train_performance.sh --data_path=<imagenet_path> --rank_size=4
    ```
    更多训练参数参考 train.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./test/loss.py)）:
MeanRelativeError: -0.1429139638086771
MeanAbsoluteError: -2.5690939999999993
Rule,mean_absolute_error -2.5690939999999993
pass mean_relative_error=-0.1429139638086771 <= 0.05 or mean_absolute_error=-2.5690939999999993 <= 0.0002




