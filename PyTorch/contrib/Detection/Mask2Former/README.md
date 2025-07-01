# Mask2former
## 1. 模型概述
Mask2former 是一种基于全新 mask 解码器和多尺度特征融合的统一分割框架，能够实现语义分割、实例分割和全景分割任务。

- 论文链接：https://arxiv.org/abs/2112.01527
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/mask2former

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
<MODLE Mask2former>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。


#### 2.2.2 处理数据集
1.具体配置方式可参考：https://github.com/Atten4Vis/ConditionalDETR/blob/main/README.md。
2.或者按下述：
datasets #根目录
  /coco
    /annotations
      /panoptic_train2017.json
      /panoptic_val2017.json
      /panoptic_train2017 #图像分割训练集
      /panoptic_val2017
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
  cd <ModelZoo_path>/PyTorch/contrib/Detection/Mask2former/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_Mask2former.py --config ../configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py --launcher pytorch --nproc-per-node 1  --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/" 
  ```
  # 该模型开启amp在sdaa和cuda上都会梯度爆炸，长nan现象，且训练很慢
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.0024590707799176438
MeanAbsoluteError: -4.644323873519897
Rule,mean_absolute_error -4.644323873519897
pass mean_relative_error=-0.0024590707799176438 <= 0.05 or mean_absolute_error=-4.644323873519897 <= 0.0002