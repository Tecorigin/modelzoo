# YOLOv11

## 1.模型概述
YOLOv11 是一种高效的目标检测模型，基于 YOLO（You Only Look Once）系列的最新改进版本。它通过优化网络结构和训练策略，进一步提升了检测精度和推理速度。YOLOv11 引入了多尺度特征融合和自适应锚框机制，增强了模型对不同尺度目标的检测能力，同时减少了计算复杂度。

YOLOv11 的亮点在于其平衡了精度与速度，适用于实时目标检测任务，如自动驾驶、视频监控等。尽管模型结构复杂，但 YOLOv11 在多个基准数据集上表现优异，为目标检测领域提供了新的解决方案。YOLOv11 的出现推动了实时检测技术的发展，为高精度、高效率的目标检测任务提供了重要参考。其设计思想也影响了后续检测模型的优化方向，成为目标检测领域的经典代表之一。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

YOLOv11运行在COCO数据集上，数据集获取可以参考如下说明：
```
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```
为了将数据从91类COCO格式处理为80类YOLO格式，可参考scripts/process_data.sh中的操作，使用coco2yolo.py脚本来处理数据标签并整理数据目录结构：
```shell
python coco2yolo.py

cd /path/to/coco
mv convert/images .
mv convert/labels .
mv train2017 images
mv val2017 images
```

### 2.3 启动训练

准备环境（在模型主目录下）：
```
pip install -r requirements.txt
pip install -e .
```

模型单机单卡训练：
```
export SDAA_VISIBLE_DEVICES=0,1,2,3

python train.py 2>&1|tee scripts/train_sdaa_3rd_test.log
```

### 2.4 训练结果

模型训练2小时后，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train box Loss | Train cls Loss | Train dfl Loss | AccTop1 |
|---|---|---|---|---|---|---|---|---|
| 1 | YOLOv11 | Amp | 16 | Epoch：2 | 1.343 | 1.821 | 1.361 | / |