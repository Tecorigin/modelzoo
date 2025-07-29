# EncNet
## 1. 模型概述
本文针对场景分割任务，提出基于自注意力机制的全局上下文依赖建模方法。不同于以往通过多尺度特征融合捕获上下文的工作，我们设计了一种双注意力网络（DANet），能够自适应地整合局部特征与其全局依赖关系。具体而言，我们在传统空洞卷积FCN基础上增加了两种注意力模块：空间维度与通道维度的语义互依赖建模模块。其中，位置注意力模块通过加权聚合所有位置的特征实现选择性特征整合，使语义相似的特征建立关联（无论其空间距离远近）；通道注意力模块则通过关联所有通道图的特征，选择性强化相互依赖的特征通道。我们将两个模块的输出特征相加，进一步增强特征表示能力，从而获得更精确的分割结果。
- 论文链接：[Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983)
- 仓库链接：[https://github.com/open-mmlab/mmsegmentation/tree/main/configs/danet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/danet)

1.基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2.获取数据集：介绍如何获取训练所需的数据集。
3.构建环境：介绍如何构建模型运行所需要的环境。
4.启动训练：介绍如何运行训练。

## 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。
## 2.2 准备数据集
DaNet使用Cityscapes数据集，该数据集为开源数据集，可从[CityScapes](https://www.cityscapes-dataset.com/login/)下载。
## 2.3 构建环境
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
## 2.4 启动训练

1.在构建好的环境中，进入训练脚本所在目录。
   ```
   cd <ModelZoo_path>/PyTorch/contrib/Segmentation/DaNet/run_scripts
   ``` 
2. 运行训练。该模型支持单机单卡。
   ```
   python run_danet.py --config ../configs/danet/danet_r50-d8_4xb2-80k_cityscapes-512x1024.py \
    --launcher pytorch --nproc-per-node 1 --amp 2>&1 | tee sdaa.log
   ```
更多训练参数参考 run_scripts/argument.py

## 2.5 训练结果
输出训练loss曲线及结果:

![loss](./run_scripts/loss.jpg)

MeanRelativeError: -0.38333116476042733

MeanAbsoluteError: -2.325974152088165

Rule,mean_absolute_error -2.325974152088165

pass mean_relative_error=-0.38333116476042733 <= 0.05 or mean_absolute_error=-2.325974152088165 <= 0.0002