
# Vgg
## 1. 模型概述
Vgg(Visual Geometry Group Network)是由牛津大学视觉几何组（Visual Geometry Group）提出的一种深度卷积神经网络（CNN），其核心思想是通过堆叠多个3×3小卷积核来构建深层网络，证明了网络深度对模型性能的重要性。VGG在2014年ImageNet图像分类竞赛（ILSVRC）中取得了优异成绩，成为深度学习领域的重要基准模型之一。

- 论文链接：[[1409.1556v6\]]Very Deep Convolutional Networks for Large-Scale Image Recognition(https://arxiv.org/abs/1409.1556)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/vgg

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
Vgg 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Classification/vgg/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_vgg.py --config ../configs/vgg/vgg11_8xb32_in1k.py --launcher pytorch --nproc-per-node 4 --amp --cfg-options "train_dataloader.dataset.data_root=<imagenet_path>" "val_dataloader.dataset.data_root=<imagenet_path>"
  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![vgg_loss_compare](./images/vgg_loss_compare.jpg)

MeanRelativeError: 0.0008037261371356254
MeanAbsoluteError: 0.005479326342592145
Rule,mean_relative_error 0.0008037261371356254
pass mean_relative_error=0.0008037261371356254 <= 0.05 or mean_absolute_error=0.005479326342592145 <= 0.0002