#  Yolov9-c

## 1. 模型概述
YOLOv9 是 YOLOv7 研究团队推出的目标检测网络，它是 YOLO（You Only Look Once）系列的迭代。YOLOv9 在设计上旨在解决深度学习中信息瓶颈问题，并提高模型在不同任务上的准确性和参数效率

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 环境准备

#### 2.1.1 拉取代码仓

``` bash
git clone https://gitee.com/tecorigin/modelzoo.git
```

#### 2.1.2 Docker 环境准备

##### 获取 SDAA Pytorch 基础 Docker 环境

SDAA 提供了支持 Pytorch 的 Docker 镜像，请参考 [Teco文档中心的教程](http://docs.tecorigin.com/release/tecopytorch/v1.5.0/) -> 安装指南 -> Docker安装 中的内容进行 SDAA Pytorch 基础 Docker 镜像的部署。

##### 激活 Teco Pytorch 虚拟环境
使用如下命令激活并验证 torch_env 环境

``` bash
conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"

--------------+----------------------------------------------
Host IP      | 172.17.0.8
PyTorch      | 2.0.0a0+gitdfe6533
Torch-SDAA   | 1.7.0
--------------+----------------------------------------------
SDAA Driver  | 1.2.0 (N/A)
SDAA Runtime | 1.2.0 (/opt/tecoai/lib64/libsdaart.so)
SDPTI        | 1.1.1 (/opt/tecoai/lib64/libsdpti.so)
TecoDNN      | 1.20.0 (/opt/tecoai/lib64/libtecodnn.so)
TecoBLAS     | 1.20.0 (/opt/tecoai/lib64/libtecoblas.so)
CustomDNN    | 1.20.0 (/opt/tecoai/lib64/libtecodnn_ext.so)
TecoRAND     | 1.7.0 (/opt/tecoai/lib64/libtecorand.so)
TCCL         | 1.17.0 (/opt/tecoai/lib64/libtccl.so)
--------------+----------------------------------------------
```

##### 2.1.3 安装依赖模块
使用如下命令安装依赖模块

``` bash
pip install -r requirements.txt
```

### 2.2 数据集准备

进行数据集的下载与准备(百度网盘形式但需要下载到yolov9c文件夹中)
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Detection/yolov9c
百度网盘链接:https://pan.baidu.com/s/1j5NOfgFcxCyS2je3fNLekA?pwd=5qx5
其中coco.yaml中前三行需要修改为：
train: <ModelZoo_path>/PyTorch/contrib/Detection/yolov9c/data/images/train
val: <ModelZoo_path>/PyTorch/contrib/Detection/yolov9c/data/images/val
test: <ModelZoo_path>/PyTorch/contrib/Detection/yolov9c/data/images/val
```
```
目录结构如下

└── data
    ├──hyps  
    ├──images
    │   ├──train
    │   └──val  
    ├──labels
    │   ├──train
    │   └──val
    └── coco.yaml
```

    
### 2.3 启动训练

运行示例
下面给出了一个训练模型的示例脚本。

#### 2.3.1 在构建好的环境中，进入训练脚本所在目录。
   ```
    cd <ModelZoo_path>/PyTorch/contrib/Detection/yolov9c
   ```
   
#### 2.3.2 运行训练。该模型支持单机单卡。

##### 训练

- 单机单卡
   ```
   python train_dual.py --cfg ./models/detect/yolov9-c.yaml --data ./data/coco.yaml --hyp ./data/hyps/hyp.scratch-high.yaml --batch-size 8 --epochs 2 --device sdaa 
   ```
其他参数请见[参数参考](./run_scripts/参数参考.md)
##### 测试

- 单机单卡
  ```
  python val_dual.py --data ./data/coco.yaml --weights ./runs/train/exp2/weights/best.pt --batch_size 32 --device sdaa
  ```


## 3 训练条件
| 设备名称  | 模型名称    | epoch | batch-size | 数据集     | 混合精度 | MAP  |
|-------|---------|-------|------------|---------|------|------|
| teco6 | yolov9c | 2     | 8          | voc2012 | 是    | 0.05 |

注:yolov9c训练时间过长，在于管理人员商量后仅完成了适配，因此只跑了一个epoch并进行测试，效果并未到达理想状态。

