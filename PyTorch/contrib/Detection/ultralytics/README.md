# Ultralytics YOLO
## 介绍

* Ultralytics YOLO 基于深度学习和计算机视觉领域的尖端技术，在速度和准确性方面具有无与伦比的性能。其流线型设计使其适用于各种应用，并可轻松适应从边缘设备到云 API 等不同硬件平台。

## 特征

* 全流程适配：训练、验证、推理全流程适配

* 模型覆盖全面：支持 YOLOv3、YOLOv5、YOLOv6、YOLOv8、YOLOv9、YOLO-World、RT-DETR 等多种模型

* 任务支持广泛：支持目标检测、语义分割、图像分类、姿态识别、OBB 检测等多种任务

* 数据集支持广泛：支持 COCO、VOC、ImageNet 等多种数据集

* 高性能：支持多卡训练，支持 AMP 混合精度训练

* 支持预训练模型：兼容官方预训练模型，可快速进行微调训练


## 模型支持
* 以下是经过测试已支持的主要模型：

    | 模型                                          | 简介                                                                                                                | 支持任务 | 自测结果（African-wildlif @ 5 epochs） |AMP| Batch Size|
    |------------------------------------------------|----------------------------------------------------------------------------------|--------|--------|--------|--------|
    | YOLOv3                                         | YOLO 模型系列的第三次迭代，最初由 Joseph Redmon 设计，以其高效的实时物体检测功能而闻名。                                     | 目标检测 | yolov3u (mAP50 = 0.809) | 启用| 12 |
    | YOLOv5                                         | Ultralytics 的 YOLO 架构的改进版本，与以前的版本相比，性能和速度都有所提高。                                               | 目标检测 | yolov5nu (mAP50 = 0.922) | 启用 | 12 |
    | YOLOv8                                  | YOLO 系列的最新版本，具有实例分割、姿势/关键点估计和分类等增强功能。                                                   | 目标检测 / 语义分割 / 图像分类 / 姿态识别 / OBB 检测 | yolov8n (mAP50 = 0.923) | 启用 | 12 |
    | YOLOv9                                         | 实验模型 Ultralytics YOLOv5 实现可编程梯度信息 (PGI) 的代码库。                                                            | 目标检测 / 语义分割 | yolov9t (mAP50 = 0.919)| 启用 | 12 |
    | YOLOv10| YOLOv10 是清华大学研究人员在 Ultralytics YOLO 基础上引入的一种新的实时目标检测方法。|  目标检测 | yolov10n (mAP50 = 0.811) | 启用 | 12 |
    | RT-DETR |  由百度公司开发的实时检测模型在保持高精度的同时提供实时性能。 | 目标检测 | rtdetr-l (mAP50 = 0.278)| 关闭 | 12 |
    | YOLO-World | 基于 Ultralytics YOLOv8 和 CLIP 开发的用于开放词汇检测任务的实时检测模型。 |  目标检测 | yolov8s-world (mAP50 = 0.912)| 关闭 | 12 |

## 快速指南

### 1、环境准备

#### 1.1 拉取代码仓

* 克隆模型库代码

    ``` bash
    git clone https://gitee.com/tecorigin/modelzoo.git
    ```

#### 1.2 Docker 环境准备

##### 1.2.1 获取 SDAA Pytorch 基础 Docker 环境

* SDAA 提供了支持 Pytorch 的 Docker 镜像，请参考 [Teco文档中心的教程](http://docs.tecorigin.com/release/tecopytorch/v1.5.0/) -> 安装指南 -> Docker安装 中的内容进行 SDAA Pytorch 基础 Docker 镜像的部署。

##### 1.2.2 激活 Teco Pytorch 虚拟环境
* 使用如下命令激活并验证 torch_env 环境

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
* 使用如下命令安装依赖模块
    
    ``` bash
    # 切换到工作目录
    cd PyTorch/contrib/Detection/ultralytics
    
    # 安装 Ultralytics
    pip install .
    
    # 安装 Tcap DLloger
    pip install git+https://gitee.com/xiwei777/tcap_dllogger.git
    ```
    
### 2、数据集准备
#### 2.1 获取数据集
* 训练时会自动下载所需要的数据集，如 COCO、VOC、African-wildlife 等，更多数据集详情请参考 [Ultralytics 数据集文档](https://docs.ultralytics.com/datasets)

#### 2.2 数据集目录结构

* 这里以示例的 African-wildlife 数据集为例：

    ```
    └── african-wildlife
        ├──train
        │   ├──images
        │   │   ├──图片1
        │   │   ├──图片2
        │   │   └── ...
        │   ├──labels
        │   │   ├──标注1
        │   │   ├──标注2
        │   │   └── ...
        ├──valid
        │   ├──images
        │   │   ├──图片1
        │   │   ├──图片2
        │   │   └── ...
        │   ├──labels
        │   │   ├──标注1
        │   │   ├──标注2
        │   │   └── ...
        ├──test
        │   ├──images
        │   │   ├──图片1
        │   │   ├──图片2
        │   │   └── ...
        │   ├──labels
        │   │   ├──标注1
        │   │   ├──标注2
        │   │   └── ...
    ```
    

### 3、African-wildlife 自测

#### 3.1 训练示例

* 下面给出了一个使用 African-wildlife 数据集训练 YOLOv8n 模型的示例脚本（单卡三核组），更多的训练参数可以参考 [run_scripts/README.md](./run_scripts/README.md)。

    ```bash
    python run_scripts/run_train_yolo.py --mode=detect --data=african-wildlife.yaml --model=yolov8n.pt --epochs=5 --device=[0,1,2] --batch=12
    ```

### 4. 模型精度验证

#### 4.1 验证示例

* 使用下面的命令加载训练完成的模型进行精度验证

    ```bash
    yolo detect val data=african-wildlife.yaml model=runs/detect/train/weights/best.pt
    ```
 

#### 4.2 自测精度

* 在 African-wildlife 数据集下的自测精度如下：

    | Class       | Images | Instances | Box(P) | R      | mAP50  | mAP50-95 |
    |-------------|--------|-----------|--------|--------|--------|----------|
    | all         | 225    | 379       | 0.92607  | 0.82407  | 0.9225  | 0.73009    |
    
