# EncNet
## 1. 模型概述
近期研究通过采用空洞卷积、利用多尺度特征及优化边界处理，在全卷积网络（FCN）框架下显著提升了像素级标注的空间分辨率。本文通过引入上下文编码模块，探索全局语境信息对语义分割的影响——该模块能捕捉场景的语义上下文，并选择性增强类别相关特征图。所提出的上下文编码模块仅需在FCN基础上增加少量计算成本，即可显著改善语义分割效果：我们在PASCAL-Context数据集上取得51.7% mIoU，在PASCAL VOC 2012数据集上达到85.9% mIoU的新标杆。单个模型在ADE20K测试集上最终得分为0.5567，超越了2017年COCO-Place挑战赛的冠军方案。此外，我们还探究了该模块如何提升浅层网络在CIFAR-10图像分类任务中的特征表征能力——仅14层的网络实现了3.45%的错误率，与超过其10倍深度的最先进模型性能相当。
- 论文链接：[Context Encoding for Semantic Segmentation](https://arxiv.org/abs/1803.08904)
- 仓库链接：[Code](https://github.com/zhanghang1989/PyTorch-Encoding)
## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 在构建好的环境中，进入训练脚本所在目录。
   ```
   cd <ModelZoo_path>/PyTorch/contrib/Segmentation/encnet/run_scripts
   ```
2. 运行训练。该模型支持单机单卡。
   ```
   python run_encnet.py --config ../configs/encnet/encnet_r50-d8_4xb2-80k_cityscapes-512x1024.py \
    --launcher pytorch --nproc-per-node 1 --amp 2>&1 | tee sdaa.log
   ```
   
![loss](./run_scripts/loss.jpg)
MeanRelativeError:-0.19116692150429956

MeanAbsoluteError:-0.49424196600914

Rule,mean absolute error=-0.49424196600914

pass mean relative error=-0.19116692150429956 <= 0.05 or mean absolute error=-0.49424196600914 <= 0.0002