 # Faster_rcnn：目标检测模型在Pytorch当中的实现
## 1、模型概述
faster_rcnn是一种用于目标检测的深度学习模型，由Microsoft Research在2015年提出，它在R-CNN和Fast R-CNN的基础上做出了进一步的改进，主要解决了目标检测中的几个关键问题：检测速度慢和训练过程复杂。
## 2、快速开始
使用本模型执行训练的主要流程如下：

1.运行环境配置：介绍训练前需要完成的运行环境配置和检查。

2.数据集准备：介绍如何使用如何获取并处理数据集。

3.启动训练：介绍如何运行训练。
### 2.1、运行环境配置
#### 2.1.1 拉取代码仓
```python
git clone https://gitee.com/tecorigin/modelzoo.git
```

#### 2.1.2 创建Teco虚拟环境
```python
cd /modelzoo/PyTorch/contrib/faster_net
conda activate torch_env

pip install -r requirements.txt
```
### 2.2、数据集准备
#### 2.2.1 数据集介绍
VOC数据集是一个世界级的计算机视觉挑战赛PASCAL VOC挑战赛所提出的数据集，用于分类和检测的数据集规模为：train/val ：11540 张图片，包含 27450 个已被标注的 ROI annotated objects ；用于分割的数据集规模为：trainval：2913张图片，6929个分割，
#### 2.2.2 数据集下载
VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/1-1Ej6dayrx3g0iAA88uY5A    
提取码: ph32
#### 2.2.3 数据集处理
```python
python voc_annotation.py
# 利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt
```
### 2.3、预训练权重
#### 2.3.1 权重下载
下载链接如下：https://pan.baidu.com/s/1S6wG8sEXBeoSec95NZxmlQ
提取码: 8mgp
下载其中的voc_weights_vgg.pth和vgg16-397923af.pth即可
#### 2.3.2 权重位置
请将pth文件放置在model_data文件夹下

### 2.4、启动训练
#### 2.4.1、训练命令
支持单机单SPA以及单机单卡（DDP）。训练过程保存的权重以及日志均会保存在"logs"中。\
单机单SPA（单精度）
```python
python run_scripts/run_faster_rcnn.py  --model_name=faster_net  --epoch=10  --batch_size=8  --nproc_per_node=1  --device=sdaa  --use_amp=False  --use_ddp=False  --classes_path='model_data/voc_classes.txt'
```
单机单SPA（混合精度）
```python
python run_scripts/run_faster_rcnn.py  --model_name=faster_net  --epoch=10  --batch_size=8  --nproc_per_node=1  --device=sdaa  --use_amp=True  --use_ddp=False   --classes_path='model_data/voc_classes.txt'
```
单机单卡(DDP)（混合精度）
```python
python run_scripts/run_faster_rcnn.py  --model_name=faster_net  --epoch=10  --batch_size=8      --nproc_per_node 4  --device=sdaa  --use_amp=True  --use_ddp=True   --classes_path='model_data/voc_classes.txt'
```
#### 2.4.2、测试命令
```python
python get_mAP.py  --model_path=logs/last_epoch_weights.pth  --classes_path=model_data/voc_classes.txt  --dataset_path=VOCdevkit
```
### 2.5 训练结果
训练条件

| 芯片   | 卡 | 模型          | 混合精度 | batch size |
|------|---|-------------|------|------------|
| SDAA | 1 | faster_rcnn | 是    | 8          |

训练结果量化指标如下表所示

| 加速卡数量 | epoch | 混合精度 | batch size | mAP(%) |
|-------|-------|------|------------|---------|
| 1     | 10    | 是    | 8          | 84.99  |


