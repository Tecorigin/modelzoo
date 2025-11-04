# DINO
## 1. 模型概述
DINO (DETR with Improved DeNoising Anchor Boxes) 是一款在 ICLR 2023 发布的前沿端到端目标检测 Transformer 模型,是
首个在 COCO leader 上夺魁的 end-to-end Transformer 检测模型，通过噪声对比训练+动态 anchor queries+双层梯度 lo‑forward，
兼顾速度与精度，为 DETR 设立新标杆。

- 论文链接：https://arxiv.org/abs/2203.03605
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/dino

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
<MODLE DINO>使用 COCO2017 数据集，该数据集为开源数据集，可从 [COCO](https://cocodataset.org/#download) 下载。

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
  cd <ModelZoo_path>/PyTorch/contrib/Detection/DINO/run_scripts
  ```
2. 运行训练。该模型支持单机单卡。
  ```
  python run_DINO.py --config ../configs/dino/dino-4scale_r50_8xb2-36e_coco.py --launcher pytorch --nproc-per-node 1 --amp --cfg-options "train_dataloader.dataset.data_root=/data/teco-data/coco/" "val_dataloader.dataset.data_root=/data/teco-data/coco/"
  ```
    更多训练参数参考 run_scripts/argument.py
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.019506659922123023
MeanAbsoluteError: -4.331806333065033
Rule,mean_absolute_error -4.331806333065033
pass mean_relative_error=-0.019506659922123023 <= 0.05 or mean_absolute_error=-4.331806333065033 <= 0.0002