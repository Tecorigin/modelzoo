
# Vgg
## 1. 模型概述
CAE（Computer-Aided Engineering）模型是一种基于计算机仿真技术，对产品在结构、热力、流体、电磁等工况下性能进行预测与优化的数字化分析模型。

- 论文链接：[[arXiv:2202.03026]]Context Autoencoder for Self-Supervised Representation Learning(https://arxiv.org/abs/2202.030266)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/cae

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/cae/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   torchrun --master_port=29500 ./run_cae.py ./cae/cae_beit-base-p16_8xb256-amp-coslr-300e_in1k --launcher pytorch --amp | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 



MeanRelativeError: -0.0027313013092655825
MeanAbsoluteError: -0.03529393196105957
Rule,mean_absolute_error -0.03529393196105957
pass mean_relative_error=-0.0027313013092655825 <= 0.05 or mean_absolute_error=-0.03529393196105957 <= 0.0002