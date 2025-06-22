# EfficientNet
## 1. 模型概述
EfficientNet 是由 Google Research 在 2019 年提出的一系列高效卷积神经网络（CNN），其核心思想是通过复合缩放（Compound Scaling）策略，在计算资源受限的情况下，实现更高的准确率和更低的计算成本。EfficientNet 在 ImageNet 分类任务上表现优异，并在多个视觉任务（如目标检测、语义分割）中成为基准模型之一。

- 论文链接：[[1905.11946\]]EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks(https://arxiv.org/abs/1905.11946v5)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/efficientnet

使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
EfficientNet使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/efficient_net/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python run_efficient_net.py --config ../configs/efficient_net/efficientnet-b0_8xb32_in1k.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.00031475151136514386
MeanAbsoluteError: -0.002962395696356745
Rule,mean_absolute_error -0.002962395696356745
pass mean_relative_error=-0.00031475151136514386 <= 0.05 or mean_absolute_error=-0.002962395696356745 <= 0.0002