# ShuffleNet_v1
## 1. 模型概述
ShuffleNet v1(An Extremely Efficient Convolutional Neural Network for Mobile Devices)是由旷视科技(Megvii)的研究团队于2017年提出的一种专为移动设备设计的高效卷积神经网络架构。该模型主要针对计算资源受限的环境(如移动端和嵌入式设备)，在保持较高精度的同时大幅降低计算复杂度。

- 论文链接：[[1707.01083\]]An Extremely Efficient Convolutional Neural Network for Mobile Devices(https://arxiv.org/abs/1707.01083v2)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/shufflenet_v1

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
    git clone https://gitee.com/xiwei777/mmengine_sdaa.git 
    cd mmengine_sdaa 
    pip3 install -r requirements.txt 
    pip3 install opencv_python mmcv --no-deps
    python setup.py install 
    cd .. 
    git clone http://10.10.30.109/tecoap1/application/mmpretrain.git 
    pip install -r requirements.txt
    pip install -e .
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ShuffleNet_v1/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --master_port=29500 ./run_shufflenet_v1.py ./shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py --launcher pytorch --amp | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![shufflenet_v1_loss_compare](./image/shufflenet_v1_loss_compare.jpg)

MeanRelativeError: -0.031109087576603318
MeanAbsoluteError: -0.5176715661983678
Rule,mean_absolute_error -0.5176715661983678
pass mean_relative_error=-0.031109087576603318 <= 0.05 or mean_absolute_error=-0.5176715661983678 <= 0.0002