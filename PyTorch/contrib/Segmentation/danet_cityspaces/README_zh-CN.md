# EncNet
## 1. 模型概述
本文针对场景分割任务，提出基于自注意力机制的全局上下文依赖建模方法。不同于以往通过多尺度特征融合捕获上下文的工作，我们设计了一种双注意力网络（DANet），能够自适应地整合局部特征与其全局依赖关系。具体而言，我们在传统空洞卷积FCN基础上增加了两种注意力模块：空间维度与通道维度的语义互依赖建模模块。其中，位置注意力模块通过加权聚合所有位置的特征实现选择性特征整合，使语义相似的特征建立关联（无论其空间距离远近）；通道注意力模块则通过关联所有通道图的特征，选择性强化相互依赖的特征通道。我们将两个模块的输出特征相加，进一步增强特征表示能力，从而获得更精确的分割结果。在Cityscapes、PASCAL Context和COCO Stuff三个具有挑战性的场景分割数据集上，我们的方法均取得了最先进的性能表现——特别地，在不使用粗标注数据的条件下，我们在Cityscapes测试集上实现了81.5%的平均IoU分数。
- 论文链接：[Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983)
- 仓库链接：[Code](https://github.com/junfu1115/DANet/)
## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 在构建好的环境中，进入训练脚本所在目录。
   ```
   cd <ModelZoo_path>/PyTorch/contrib/Segmentation/danet_cityspaces/run_scripts
   ```
2. 运行训练。该模型支持单机单卡。
   ```
   python run_danet.py --config ../configs/danet/danet_r50-d8_4xb2-80k_cityscapes-512x1024.py \
    --launcher pytorch --nproc-per-node 1 --amp 2>&1 | tee sdaa.log
   ```
   
![loss](./run_scripts/loss.jpg)

MeanRelativeError: -0.38333116476042733

MeanAbsoluteError: -2.325974152088165

Rule,mean_absolute_error -2.325974152088165

pass mean_relative_error=-0.38333116476042733 <= 0.05 or mean_absolute_error=-2.325974152088165 <= 0.0002
