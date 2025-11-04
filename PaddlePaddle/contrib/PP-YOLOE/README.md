# PP-YOLOE

## 1.模型概述
PP-YOLOE 是一种高效的目标检测模型，专为实际应用场景中的速度与精度平衡而设计。它通过改进的网络架构和优化的训练策略，在保持高检测精度的同时，显著提升了模型的推理速度。PP-YOLOE 的核心优势在于其高效的特征提取机制和改进的锚点分配策略，能够在复杂场景下快速准确地检测目标。

PP-YOLOE 在多个基准数据集上展现了卓越的性能，尤其在处理大规模数据集时表现出色。其创新的架构设计使得模型在训练阶段能够更好地学习特征，在推理阶段则保持高效运行。PP-YOLOE 的出现为实时目标检测任务提供了新的高效解决方案，推动了目标检测技术在工业、安防等领域的广泛应用。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

PP-YOLOE运行在COCO2017数据集上，数据集配置可以参考:
```
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### 2.3 启动训练

模型单机单卡训练：
```shell
export PADDLE_XCCL_BACKEND=sdaa
export PADDLE_DISTRI_BACKEND=tccl
export SDAA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml --eval --amp 2>&1|tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练总耗时为2小时，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss |
|---|---|---|---|---|---|
| 1 | PP-YOLOE | Amp | 8 | Iterations:2900/3664 | / |

该模型训练过程loss持续爆炸，相同代码在该模型的单机单核组训练过程中和PP-PicoDet模型的单机单卡训练过程中都表现出正常loss数值范围。