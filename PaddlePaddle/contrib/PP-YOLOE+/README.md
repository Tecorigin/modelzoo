# PP-YOLOE+

## 1.模型概述
PP-YOLOE+ 是一种高效且高精度的目标检测模型，基于 PP-YOLOE 进一步优化，旨在实现精度与效率的平衡。它采用无锚点的检测机制，结合改进的骨干网络和颈部结构，增强了特征提取能力。此外，PP-YOLOE+ 引入了更高效的标签分配策略（如任务对齐学习 TAL）和改进的 ET-Head（Efficient Task-aligned Head），通过优化损失函数（如 VariFocal Loss 和 Distribution Focal Loss）进一步提升检测精度。

PP-YOLOE+ 在 COCO 数据集上表现出色，mAP 分数显著提升，同时保持较高的推理速度。其设计注重在复杂场景下的稳健性，尤其在小目标检测和高分辨率图像处理方面表现出色。PP-YOLOE+ 的推出为需要高精度的目标检测任务提供了强大的技术支持，适用于安防监控、自动驾驶等领域。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

PP-YOLOE+运行在COCO2017数据集上，数据集配置可以参考:
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
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml --eval --amp 2>&1|tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练总耗时为2小时，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss |
|---|---|---|---|---|---|
| 1 | PP-YOLOE+ | Amp | 8 | Epochs:3 | / |

该模型训练过程loss持续爆炸，相同代码在该模型的单机单核组训练过程中和PP-PicoDet模型的单机单卡训练过程中都表现出正常loss数值范围。