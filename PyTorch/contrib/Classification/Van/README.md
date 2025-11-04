# VAN
## 1. 模型概述
VAN（Visual Attention Network）是一种基于视觉注意力机制的深度学习模型，旨在高效处理图像识别、目标检测、语义分割等计算机视觉任务。VAN通过结合局部感知与全局注意力机制，显著提升了模型对图像特征的提取能力，同时保持了较高的计算效率。

- 论文链接：[[2202.09741]]]Visual Attention Network(https://arxiv.org/abs/2202.09741)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/van
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
VAN使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/Van/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python run_van.py --config ../configs/van/van-tiny_8xb128_in1k.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.00013675602948527824
MeanAbsoluteError: -0.0009605105560604888
Rule,mean_absolute_error -0.0009605105560604888
pass mean_relative_error=-0.00013675602948527824 <= 0.05 or mean_absolute_error=-0.0009605105560604888 <= 0.0002