# XCiT
## 1. 模型概述
XCiT 是一种创新的视觉Transformer架构，由Facebook AI Research于2021年提出，旨在解决传统自注意力机制在视觉任务中的计算效率问题。它通过交叉协方差注意力（Cross-Covariance Attention, XCA）替代标准自注意力，显著提升了长序列处理效率，同时保持优异的性能。

- 论文链接：[[2106.09681\]]XCiT: Cross-Covariance Image Transformers(https://arxiv.org/abs/2106.09681)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/xcit
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
XCiT使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/XCiT/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python run_xcit.py --config ../configs/xcit/xcit-nano-12-p16_8xb128_in1k.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.00012661599296072512
MeanAbsoluteError: -0.0008858784590617264
Rule,mean_absolute_error -0.0008858784590617264
pass mean_relative_error=-0.00012661599296072512 <= 0.05 or mean_absolute_error=-0.0008858784590617264 <= 0.0002