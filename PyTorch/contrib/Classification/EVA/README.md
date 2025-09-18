# EVA
## 1. 模型概述
EVA是一个高性能的开源视觉基础模型，它通过以CLIP为锚点进行改进，在多个视觉及多模态任务上取得了顶尖性能。

- 论文链接：[[2211.07636]]EVA: Exploring the Limits of Masked Visual Representation Learning at Scale(https://arxiv.org/abs/2211.07636)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/eva


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
EVA使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/EVA/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
python run_eva.py --config ../configs/eva/eva-l-p14_8xb16_in1k-196px.py --launcher pytorch --nproc-per-node 1 --amp | tee sdaa.log

   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: 0.008249334472737697
MeanAbsoluteError: 0.040366385242726545
Rule,mean_relative_error 0.008249334472737697
pass mean_relative_error=0.008249334472737697 <= 0.05 or mean_absolute_error=0.040366385242726545 <= 0.0002