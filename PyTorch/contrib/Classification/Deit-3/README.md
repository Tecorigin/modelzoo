# DeiT-3
## 1. 模型概述
DeiT-III 是DeiT系列的最新版本（2022年提出），专注于 进一步提升纯视觉Transformer（ViT）在ImageNet-1K等中小型数据集上的性能，无需依赖大规模预训练数据。其核心思想是通过架构简化和训练策略优化，挖掘ViT在数据效率与模型性能上的潜力。

- 论文链接：[[2204.07118]]MLP-Mixer: An all-MLP Architecture for Vision(https://arxiv.org/abs/2204.07118)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/deit3
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
DeiT-3使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/Deit-3/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python run_deit_3.py --config ../configs/deit_3/deit3-small-p16_64xb64_in1k.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -2.3852678978346546e-07
MeanAbsoluteError: -1.6476848337909963e-06
Rule,mean_absolute_error -1.6476848337909963e-06
pass mean_relative_error=-2.3852678978346546e-07 <= 0.05 or mean_absolute_error=-1.6476848337909963e-06 <= 0.0002