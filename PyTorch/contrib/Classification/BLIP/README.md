# BLIP
## 1. 模型概述
BLIP 是一个开创性的视觉-语言预训练模型。它的核心思想是：统一理解与生成。与之前的模型（如 CLIP）主要擅长图文检索等理解任务不同，BLIP 在一个统一的框架内，同时精通视觉-语言理解和视觉-语言生成两大任务。它通过巧妙地设计一个多任务的模型架构和解码目标，有效地利用了带有噪声的大规模网络图文数据，实现了在多种下游任务上的卓越性能。

- 论文链接：[[2201.12086]]BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation(https://arxiv.org/abs/2201.12086)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/blip
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
BLIP使用COCO数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/) 下载。

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
    pip install "pycocoevalcap"
    ```
### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/BLIP/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python run_blip.py --config ../configs/blip/blip-base_8xb32_caption.py --launcher pytorch --nproc-per-node 1 --amp --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/train2017" "val_dataloader.dataset.data_root=/data/teco-data/coco/val2017" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.009643914356339432
MeanAbsoluteError: -0.08300133034734443
Rule,mean_absolute_error -0.08300133034734443
pass mean_relative_error=-0.009643914356339432 <= 0.05 or mean_absolute_error=-0.08300133034734443 <= 0.0002