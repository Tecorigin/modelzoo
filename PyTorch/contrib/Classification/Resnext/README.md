# ResNext
## 1. 模型概述
ResNeXt（Residual Next）是由微软研究院于2016年提出的一种卷积神经网络架构，属于ResNet（残差网络）的改进版本。其核心思想是通过引入“分组卷积”和“基数（Cardinality）”的概念，在保持模型复杂度的同时显著提升特征表达能力，同时避免了传统增加深度或宽度带来的计算成本激增问题。

- 论文链接：[[1611.05431\]]Aggregated Residual Transformations for Deep Neural Networks(https://doi.org/10.48550/arXiv.1611.05431)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/resnext

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
ResNext 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/xzxg001/article/details/142465729。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip3 install  -U openmim 
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip3 install opencv_python mmcv --no-deps
    mim install -e .
    pip install -r requirements.txt

    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
  ```
  cd <ModelZoo_path>/PyTorch/contrib/Classification/ResNext/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_resnext.py --config ../configs/resnext/resnext50-32x4d_8xb32_in1k.py --launcher pytorch --nproc-per-node 1 --amp --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/imagenet" "val_dataloader.dataset.data_root=/data/teco-data/imagenet"
  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.0006055733387035155
MeanAbsoluteError: -0.004272635620419342
Rule,mean_absolute_error -0.004272635620419342
pass mean_relative_error=-0.0006055733387035155 <= 0.05 or mean_absolute_error=-0.004272635620419342 <= 0.0002