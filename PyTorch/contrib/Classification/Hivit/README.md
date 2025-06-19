
# HiViT
## 1. 模型概述
SE-ResNet（Squeeze-and-Excitation ResNet）是由Momenta（2017）和牛津大学提出的改进版ResNet，通过引入Squeeze-and-Excitation（SE）模块，使网络能够自适应地调整通道特征的重要性，从而提升模型的表现能力。SE-ResNet在保持ResNet原有结构的基础上，显著提高了特征表示能力，并在多个视觉任务（如分类、检测、分割）上取得了更好的效果。
HiViT（Hierarchical ViT），一种全新分层视觉变换器设计，它在 MIM 中兼具高效率和良好的性能。关键在于去除不必要的“局部单元间操作”，从而得到结构简单的分层视觉变换器，其中蒙版单元可以像普通视觉变换器一样进行序列化。实证研究表明，HiViT 在全监督、自监督和迁移学习方面均表现出色。


- 论文链接：[2205.14949\]HiViT: Hierarchical Vision Transformer Meets Masked Image Modeling(https://doi.org/10.48550/arXiv.2205.14949)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/hivit

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
Deit 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/hivit/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
python run_hivit.py --config ../configs/hivit/hivit-base-p16_16xb64_in1k.py \
       --launcher pytorch --nproc-per-node 1 --amp \
       --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/imagenet" "val_dataloader.dataset.data_root=/data/teco-data/imagenet" 2>&1 | tee sdaa.log
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![loss](./image/loss.jpg)

MeanRelativeError:-0.0011513032932647313
MeanAbsoluteError:-0.008926320784162767
Rule,mean_absolute_error-0.008926320784162767
passmean_relative_error=-0.0011513032932647313 <=0.05ormean_absolute_error=-0.008926320784162767<=0.0002

