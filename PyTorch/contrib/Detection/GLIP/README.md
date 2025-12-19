# GLIP
## 1. 模型概述
是一种融合了视觉和语言信息的目标检测方法，通过引入语言提示（Prompt）提升检测模型对复杂场景和开放词汇目标的识别能力，实现了文本引导的目标检测和实例分割。

- 论文链接：https://arxiv.org/abs/2112.03857
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/glip

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
<MODLE GLIP>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

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
    pip install -r requirements/multimodal.txt
    ```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
  ```
  cd <ModelZoo_path>/PyTorch/contrib/Detection/GLIP/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  run_GLIP.py --config ../configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py --launcher pytorch --nproc-per-node 1 --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/" 
  ```
    该模型开启amp在sdaa和cuda上都会梯度爆炸，长nan现象，且训练很慢
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: 0.30944894944428797
MeanAbsoluteError: -0.10108355104923249
Rule,mean_absolute_error -0.10108355104923249
pass mean_relative_error=0.30944894944428797 <= 0.05 or mean_absolute_error=-0.10108355104923249 <= 0.0002