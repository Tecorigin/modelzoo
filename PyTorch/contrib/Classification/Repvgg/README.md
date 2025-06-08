# RepVGG
## 1. 模型概述
RepVGG 是由清华大学和旷视科技在2021年提出的高效CNN模型，其核心思想是通过结构重参数化（Structural Re-parameterization），在训练时使用多分支架构（如3x3卷积+1x1卷积+Identity分支），推理时转换为单一3x3卷积的VGG式直筒结构，实现高精度（ImageNet Top-1 80%+）与低延迟（GPU上比ResNet快）的平衡。该设计兼具训练时的多分支强表征能力和推理时的简洁高效，适用于移动端和边缘计算场景。

- 论文链接：[[2101.03697\]]RepVGG: Making VGG-style ConvNets Great Again(https://arxiv.org/abs/2101.03697)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/repvgg

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/Repvgg/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --master_port=29500 ./run_repvgg.py ./repvgg/repvgg-A0_8xb32_in1k.py --launcher pytorch --amp | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![loss_compare](./image/loss.jpg)

MeanRelativeError: 0.0017608291141246182
MeanAbsoluteError: 0.011952947862077467
Rule,mean_relative_error 0.0017608291141246182
pass mean_relative_error=0.0017608291141246182 <= 0.05 or mean_absolute_error=0.011952947862077467 <= 0.0002