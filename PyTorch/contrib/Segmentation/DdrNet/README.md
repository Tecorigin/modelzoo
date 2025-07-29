# DdrNet
## 1. 模型概述
语义分割是自动驾驶车辆理解周围环境的核心技术。当前先进模型的高性能往往依赖于高计算量和长推理时间，这在实际自动驾驶场景中难以接受。现有方法通过轻量级架构（编码器-解码器或双通路结构）或低分辨率图像推理，实现了超快的场景解析速度——在单块1080Ti GPU上甚至能超过100 FPS。然而，这些实时方法与基于空洞卷积主干网络的模型仍存在显著性能差距。为此，我们提出专为实时语义分割设计的高效主干网络家族：深度双分辨率网络（DDRNets），其核心结构包含两条深度分支，通过多级双向融合机制交互特征。此外，我们设计了新型上下文信息提取器"深度聚合金字塔池化模块"（DAPPM），基于低分辨率特征图扩大有效感受野并融合多尺度上下文信息。
- 论文链接：[Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](http://arxiv.org/abs/2101.06085)
- 仓库链接：[https://github.com/open-mmlab/mmsegmentation/tree/main/configs/ddrnet](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/ddrnet)

1.基础环境安装：介绍训练前需要完成的基础环境检查和安装。

2.获取数据集：介绍如何获取训练所需的数据集。

3.构建环境：介绍如何构建模型运行所需要的环境。

4.启动训练：介绍如何运行训练。

## 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。
## 2.2 准备数据集
DdrNet使用Cityscapes数据集，该数据集为开源数据集，可从[CityScapes](https://www.cityscapes-dataset.com/login/)下载。
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
   cd <ModelZoo_path>/PyTorch/contrib/Segmentation/DdrNet/run_scripts
   ``` 
2. 运行训练。该模型支持单机单卡。
   ```
   python run_ddrnet.py --config ../configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py \
    --launcher pytorch --nproc-per-node 1 --amp 2>&1 | tee sdaa.log
   ```
更多训练参数参考 run_scripts/argument.py

## 2.5 训练结果
输出训练loss曲线及结果:
   
![loss](./run_scripts/loss.jpg)

MeanRelativeError: -0.4552674768603906

MeanAbsoluteError: -0.9562658238410949

Rule,mean absolute error -0.9562658238410949

pass mean relative error=-0.4552674768603906 <= 0.05 or mean absolute error=-0.9562658238410949 <= 0.0002
