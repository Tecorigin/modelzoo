# DetectoRS
## 1. 模型概述
DetectoRS 是 MMDetection 提供的一种增强型两阶段目标检测器，核心创新在于引入了递归特征金字塔（Recursive Feature Pyramid, RFP）和可见性感知（Switchable Atrous Convolution, SAC）机制，从而显著提升了检测精度。

- 论文链接：https://arxiv.org/abs/2006.02334
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/detectors

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
<MODLE DetectoRS>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

#### 2.2.2 处理数据集
1.具体配置方式可参考：https://github.com/Atten4Vis/ConditionalDETR/blob/main/README.md。
2.或者按下述：
datasets #根目录
  /coco
    /annotations
    /train2017
    /val2017

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
  cd <ModelZoo_path>/PyTorch/contrib/Detection/DetectoRS/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_DetectoRS.py --config ../configs/detectors/cascade-rcnn_r50-rfp_1x_coco.py --launcher pytorch --nproc-per-node 1 --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/" 
  ```
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.022358765421009803
MeanAbsoluteError: -0.3940729123353958
Rule,mean_absolute_error -0.3940729123353958
pass mean_relative_error=-0.022358765421009803 <= 0.05 or mean_absolute_error=-0.3940729123353958 <= 0.0002