# GLIP
## 1. 模型概述
GLIP（Grounded Language-Image Pre-training）是由微软研究院（Microsoft Research）提出的多模态视觉-语言模型，其核心创新在于将目标检测（Object Detection）和视觉语言理解（Vision-Language Understanding）统一到一个框架中。GLIP 不仅能完成传统检测任务，还能实现零样本（Zero-Shot）检测和开放词汇检测（Open-Vocabulary Detection），即检测训练集中未出现过的类别。

- 论文链接：[[2112.03857\]]Grounded Language-Image Pre-training(https://arxiv.org/abs/2112.03857)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/glip
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
GLIP使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/GLIP/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python run_glip.py --config ../configs/glip/glip-l_headless.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.002793491141920383
MeanAbsoluteError: -0.021448116491336634
Rule,mean_absolute_error -0.021448116491336634
pass mean_relative_error=-0.002793491141920383 <= 0.05 or mean_absolute_error=-0.021448116491336634 <= 0.0002