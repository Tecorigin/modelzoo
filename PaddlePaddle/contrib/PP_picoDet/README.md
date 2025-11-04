# PP-PicoDet

## 1.模型概述
PP-PicoDet 是一种轻量级目标检测模型，专为移动端和边缘设备设计。它通过优化网络结构和训练策略，在保持高精度的同时显著降低了计算复杂度和模型大小。PP-PicoDet 的核心创新在于其高效的特征提取网络和轻量化的检测头，能够在资源受限的设备上实现实时目标检测。

PP-PicoDet 在多个目标检测任务中表现出色，尤其是在速度和精度的平衡上具有显著优势。其独特的架构设计使得模型在训练时能够充分学习特征，同时在推理时保持高效运行。PP-PicoDet 的出现为移动端和边缘计算场景中的目标检测提供了新的解决方案，极大地推动了深度学习在实际应用中的落地与普及。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

PP-PicoDet运行在COCO2017数据集上，数据集配置可以参考:
```
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### 2.3 启动训练

准备环境：
```
pip install -r requirements.txt
```
确保安装后环境中只有一个版本的numpy，如果不是可以在执行命令前先执行“pip uninstall -y numpy”进行卸载。

模型单机单卡训练：
```
export PADDLE_XCCL_BACKEND=sdaa
export PADDLE_DISTRI_BACKEND=tccl
export SDAA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/picodet/picodet_xs_320_coco_lcnet.yml --eval --amp 2>&1|tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练总耗时为2小时，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | PP-PicoDet | Amp | 64 | Epochs:8 | 3.2736 | / |