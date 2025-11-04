#  MMDetection
## 介绍
MMDetection 是一个基于 PyTorch 的目标检测开源工具箱。它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

## 特征
- **模块化设计**

  MMDetection 将检测框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的检测模型

- **支持多种检测任务**

  MMDetection 支持了各种不同的检测任务，包括**目标检测**，**实例分割**，**全景分割**，以及**半监督目标检测**。

  **模型支持**

  以下是支持的一些主要模型，以及简介

| 模型                                          | 简介                                                                                                                |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| albu_example                                   | 使用 Albumentations 库进行数据增强的 Mask R-CNN 模型。                                                               |
| atss                                           | 基于 ATSS (Adaptive Training Sample Selection) 算法的物体检测模型。                                                  |
| autoassign                                     | AutoAssign 模型，使用自适应分配算法进行物体检测。                                                                    |
| boxinst                                        | BoxInst 模型，一种实例分割模型，专注于高效的边界框检测。                                                              |
| carafe                                         | 使用 CARAFE 算法增强特征金字塔的 Mask R-CNN 模型。                                                                   |
| cascade_rcnn                                   | Cascade R-CNN 模型，通过多阶段检测提高精度。                                                                         |
| cascade_rpn                                    | Cascade RPN 模型，改进的区域提议网络，通过多阶段提议提高性能。                                                       |
| centernet                                      | CenterNet 模型，基于中心点检测的物体检测器。                                                                         |
| centripetalnet                                 | CentripetalNet 模型，使用 Hourglass104 网络进行高精度物体检测。                                                     |
| condinst                                       | CondInst 模型，基于条件卷积的实例分割模型。                                                                          |
| conditional_detr                               | Conditional DETR 模型，DETR 的条件变体，用于物体检测。                                                                |
| convnext                                       | 基于 ConvNeXt 变体的 Mask R-CNN 模型。                                                                               |
| cornernet                                      | CornerNet 模型，使用角点检测进行物体检测。                                                                           |
| dab_detr                                       | DAB-DETR 模型，基于动态锚框的 DEPRECATED 模型。                                                                      |
| dcn                                            | 基于可变形卷积网络 (Deformable Convolutional Networks) 的 Faster R-CNN 模型。                                          |
| dcnv2                                          | 基于可变形卷积网络 v2 的 Faster R-CNN 模型。                                                                         |
| ddod                                           | DDOD 模型，基于动态检测对象 (Dynamic Detection of Objects) 的物体检测器。                                            |
| ddq                                            | DDQ-DETR 模型，基于动态量化的 DETR 模型。                                                                            |
| deformable_detr                                | Deformable DETR 模型，DETR 的可变形卷积变体。                                                                        |
| detectors                                      | 基于 Hybrid Task Cascade (HTC) 的高级物体检测模型。                                                                  |
| detr                                           | DETR 模型，基于变压器的物体检测器。                                                                                  |
| dino                                           | DINO 模型，基于多尺度检测的物体检测器。                                                                              |
| double_heads                                   | Double-Head Faster R-CNN 模型，增强了分类和回归的检测头。                                                            |
| dsdl                                           | DSDL (Dataset Self-Labeling) 配置，用于自动化数据集标注的通用配置。                                                   |
| dyhead                                         | 使用 Dynamic Head 的 ATSS 模型，增强特征提取能力。                                                                   |
| dynamic_rcnn                                   | Dynamic R-CNN 模型，动态调整训练样本以提高检测性能。                                                                 |
| efficientnet                                   | 基于 EfficientNet 的 RetinaNet 模型，高效物体检测。                                                                  |
| empirical_attention                            | Empirical Attention Faster R-CNN 模型，使用注意力机制提升性能。                                                      |
| fast_rcnn                                      | Fast R-CNN 模型，经典的卷积神经网络物体检测器。                                                                      |
| faster_rcnn                                    | Faster R-CNN 模型，流行的卷积神经网络物体检测器。                                                                    |
| fcos                                           | FCOS 模型，基于全卷积单级物体检测器。                                                                                |
| foveabox                                       | FoveaBox 模型，一种全卷积物体检测器。                                                                                |
| fpg                                            | FPG (Feature Pyramid Grids) 模型，增强的特征金字塔网络。                                                             |
| free_anchor                                    | FreeAnchor 模型，自适应锚点分配物体检测器。                                                                          |
| fsaf                                           | FSAF (Feature Selective Anchor-Free) 模型，无锚点物体检测器。                                                        |
| gcnet                                          | GCNet 模型，使用全局上下文注意力的 Mask R-CNN 模型。                                                                |
| gfl                                            | GFL (Generalized Focal Loss) 模型，使用广义焦点损失的物体检测器。                                                    |
| ghm                                            | GHM (Gradient Harmonizing Mechanism) 模型，使用梯度协调机制的 RetinaNet。                                            |
| glip                                           | GLIP 模型，基于 ATSS 和 Swin-L 的物体检测器，适用于多尺度和大规模数据集。                                             |
| gn                                             | GN (Group Normalization) 模型，使用组归一化的 Mask R-CNN 模型。                                                     |
| gn+ws                                          | GN+WS (Weight Standardization) 模型，结合权重标准化的 Faster R-CNN。                                                  |
| grid_rcnn                                      | Grid R-CNN 模型，基于网格结构的物体检测器。                                                                          |
| groie                                          | GROIE (Generalized ROI Extraction) 模型，使用广义 ROI 提取的 Mask R-CNN。                                            |
| grounding_dino                                 | Grounding DINO 模型，基于变压器的物体检测器，适用于地面物体检测。                                                    |
| guided_anchoring                               | Guided Anchoring 模型，使用引导锚点的区域提议网络 (RPN)。                                                           |
| hrnet                                          | HRNet 模型，高分辨率网络，用于高精度物体检测。                                                                      |
| htc                                            | HTC (Hybrid Task Cascade) 模型，高级混合任务级联物体检测器。                                                        |
| instaboost                                     | InstaBoost 模型，使用 InstaBoost 数据增强技术的 Mask R-CNN。                                                        |
| lad                                            | LAD (Label Assignment Distillation) 模型，基于标签分配蒸馏的物体检测器。                                             |
| ld                                             | LD (Label Distillation) 模型，基于标签蒸馏的 GFLv1-R101 物体检测器。                                                |
| legacy_1.x                                     | 旧版本 1.x 的 Mask R-CNN 模型，经典配置。                                                                           |
| libra_rcnn                                     | Libra R-CNN 模型，使用 Libra 方法改进的 Fast R-CNN。                                                                |
| mask2former                                    | Mask2Former 模型，使用自适应遮罩生成的物体检测器。                                                                   |
| mask_rcnn                                      | Mask R-CNN 模型，广泛使用的实例分割和物体检测器。                                                                    |
| maskformer                                     | MaskFormer 模型，基于变压器的实例分割模型。                                                                          |
| misc                                           | Misc 模型，其他未分类的 Mask R-CNN 配置。                                                                            |
| ms_rcnn                                        | MS R-CNN (Mask Scoring R-CNN) 模型，基于分数的实例分割物体检测器。                                                  |
| nas_fcos                                       | NAS FCOS 模型，使用神经架构搜索 (NAS) 优化的 FCOS。                                                                 |
| nas_fpn                                        | NAS FPN 模型，使用神经架构搜索 (NAS) 优化的 RetinaNet。                                                             |
| paa                                            | PAA (Probabilistic Anchor Assignment) 模型，基于概率锚点分配的物体检测器。                                           |
| pafpn                                          | PAFPN (Path Aggregation Network) 模型，使用路径聚合网络的 Faster R-CNN。                                             |
| panoptic_fpn                                   | Panoptic FPN 模型，结合语义分割和实例分割的全景检测器。                                                             |
| pascal_voc                                     | PASCAL VOC 数据集的 Faster R-CNN 模型。                                                                             |
| pisa                                           | PISA 模型，基于概率信息选择的 RetinaNet。                                                                            |
| point_rend                                     | PointRend 模型，基于点渲染的高分辨率实例分割器。                                                                    |
| pvt                                            | PVT (Pyramid Vision Transformer) 模型，基于金字塔视觉变压器的 RetinaNet。                                            |
| queryinst                                      | QueryInst 模型，基于查询机制的实例分割模型。                                                                         |
| regnet                                         | RegNet 模型，使用 RegNetX-4GF 网络的 Mask R-CNN。                                                                   |
| reppoints                                      | RepPoints 模型，基于关键点表示的物体检测器。                                                                        |
| res2net                                        | Res2Net 模型，基于多尺度残差网络的 HTC。                                                                            |
| resnest                                        | ResNeSt 模型，使用同步 BN 的 Mask R-CNN。                                                                           |
| resnet_strikes_back                            | ResNet Strikes Back 模型，使用预训练权重的 Mask R-CNN。                                                             |
| retinanet                                      | RetinaNet 模型，高效的单级物体检测器。                                                                              |
| rpn                                            | RPN 模型，区域提议网络，用于生成候选区域。                                                                          |
| rtmdet                                         | RTMDet 模型，高效的实时物 |

体检测器。                                                                                  |
| sabl                                           | SABL (Switchable Atrous Convolution and Batch Normalization) 模型，使用可切换空洞卷积的 RetinaNet。                   |
| scnet                                          | SCNet 模型，高级物体检测模型。                                                                                      |
| scratch                                        | Scratch 模型，从头开始训练的 Mask R-CNN。                                                                            |
| selfsup_pretrain                               | Self-Supervised Pretraining 模型，使用自监督预训练的 Mask R-CNN。                                                    |
| simple_copy_paste                              | Simple Copy-Paste 模型，使用简单复制粘贴数据增强的 Mask R-CNN。                                                      |
| soft_teacher                                   | Soft Teacher 模型，使用软标签和半监督学习的 Faster R-CNN。                                                           |
| solo                                           | SOLO (Segmenting Objects by Locations) 模型，基于位置的实例分割模型。                                                |
| solov2                                         | SOLOv2 模型，改进的 SOLO 实例分割模型。                                                                             |
| sparse_rcnn                                    | Sparse R-CNN 模型，稀疏区域提议网络。                                                                               |
| ssd                                            | SSD (Single Shot MultiBox Detector) 模型，经典的单阶段物体检测器。                                                   |
| strong_baselines                               | Strong Baselines 模型，使用强基线配置的 Mask R-CNN。                                                                |
| swin                                           | Swin 模型，基于 Swin Transformer 的 Mask R-CNN。                                                                    |
| timm_example                                   | TIMM Example 模型，使用 Timm 库中预训练模型的 RetinaNet。                                                           |
| tood                                           | TOOD (Task-aligned One-stage Object Detection) 模型，任务对齐的单阶段物体检测器。                                     |
| tridentnet                                     | TridentNet 模型，基于多分支卷积的物体检测器。                                                                       |
| vfnet                                          | VFNet (VarifocalNet) 模型，使用变焦卷积的物体检测器。                                                               |
| yolact                                         | YOLACT 模型，实时实例分割器。                                                                                       |
| yolo                                           | YOLOv3 模型，YOLO 系列的第三次迭代，以高效的实时物体检测功能闻名。                                                  |
| yolof                                          | YOLOF (You Only Look One-level Feature) 模型，基于单级特征的 YOLO 变体。                                             |
| yolox                                          | YOLOX 模型，YOLO 系列的改进版本，性能和速度均有提升。                                                               |


</details>

## 快速开始

### 1 环境准备

#### 1.1 拉取代码仓

``` bash
git clone https://gitee.com/tecorigin/modelzoo.git
```

#### 1.2 Docker 环境准备

##### 1.2.1 获取 SDAA Pytorch 基础 Docker 环境

SDAA 提供了支持 Pytorch 的 Docker 镜像，请参考 [Teco文档中心的教程](http://docs.tecorigin.com/release/tecopytorch/v1.5.0/) -> 安装指南 -> Docker安装 中的内容进行 SDAA Pytorch 基础 Docker 镜像的部署。

##### 1.2.2 激活 Teco Pytorch 虚拟环境
使用如下命令激活并验证 torch_env 环境

``` bash
conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"

--------------+----------------------------------------------
Host IP      | 127.0.0.1
PyTorch      | 2.0.0a0+gitdfe6533
Torch-SDAA   | 1.5.0
--------------+----------------------------------------------
SDAA Driver  | 1.1.1 (N/A)
SDAA Runtime | 1.1.0 (/opt/tecoai/lib64/libsdaart.so)
SDPTI        | 1.1.0 (/opt/tecoai/lib64/libsdpti.so)
TecoDNN      | 1.18.0 (/opt/tecoai/lib64/libtecodnn.so)
TecoBLAS     | 1.18.0 (/opt/tecoai/lib64/libtecoblas.so)
CustomDNN    | 1.18.0 (/opt/tecoai/lib64/libtecodnn_ext.so)
TecoRAND     | 1.5.0 (/opt/tecoai/lib64/libtecorand.so)
TCCL         | 1.15.0 (/opt/tecoai/lib64/libtccl.so)
--------------+----------------------------------------------
```

##### 1.2.3 安装依赖模块
使用如下命令安装依赖模块

``` bash
pip install -r requirements/build.txt
pip install -v -e .
cd mmcv-2.1.0
pip install -v -e .
cd ..
cd mmengine
pip install -v -e .
export PYTHONPATH=$PYTHONPATH:/root/modelzoo/PyTorch/contrib/Detection/mmdetection
```

### 2 数据集准备

使用脚本自动下载，解压数据集压缩包到指定位置，里面已经包括了训练集、验证集，无需再次划分：

``` bash
python tools/misc/download_dataset.py --dataset-name voc2012 --unzip --delete
```
```
目录结构如下
├── Annotations
├── ImageSets
│   ├── Action
│   ├── Layout
│   ├── Main
│   └── Segmentation
├── JPEGImages
├── SegmentationClass
└── SegmentationObject
```

    
### 3 启动训练

运行示例
下面给出了一个训练faster_rcnn模型的示例脚本，单卡三核组。

#### 3.1 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Detection/mmdetection
    ```
   
#### 3.2. 运行训练。该模型支持单机单卡。

##### 训练

- 单机单卡四SPA
   ```
   python run_scripts/run_demo.py --model_name solov2 --nproc_per_node 4 --batch_size 4 --lr 1e-3 --epochs 10 --device sdaa --nnodes 1 --dataset_path data/coco
   ```
  
- 单机单卡单SPA
   ```
   python run_scripts/run_demo.py --model_name solov2 --nproc_per_node 1 --batch_size 4 --lr 1e-3 --epochs 10 --device sdaa --nnodes 1 --dataset_path data/voc
   ```

  更多训练参数参考[README](run_scripts/README.md)

##### 测试

- 单机单卡三SPA
  ```
  python run_scripts/test.py --model_name solov2 --nproc_per_node 4 --faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth --device sdaa --nnode 1 --node_rank 0 --data-dir data/voc
  ```

- 单机单卡单SPA
   ```
   python run_scripts/test.py --model_name solov2 --nproc_per_node 1 --faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth --device sdaa --nnode 1 --node_rank 0 --data-dir data/voc
   ```


### Reference

https://github.com/open-mmlab/mmdetection

https://mmdetection.readthedocs.io/zh-cn/v2.28.0/index.html