# YOLO12

## 1. 模型概述

YOLO12是yolo系列的经典算法，在计算机视觉领域具有广泛的应用前景。

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 数据集准备
#### 2.2.1 获取数据集


- 根据如下链接下载

[coco2017labels.zip](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip)

[train2017.zip](http://images.cocodataset.org/zips/train2017.zip)

[val2017.zip](http://images.cocodataset.org/zips/val2017.zip)

- 执行如下命令解压

```
mkdir datasets
unzip -q coco2017labels.zip -d datasets
unzip -q train2017.zip -d datasets/coco/images
unzip -q val2017.zip -d datasets/coco/images
```


#### 2.2.2 数据集目录结构参考如下所示:

```
├── datasets #根目录
  ├── coco 
      ├── annotations #json标注目录
      │   └── instances_val2017.json #对应目标检测、分割任务的验证集标注文件
      ├── images
      │   ├── train2017 #训练集图片，约118287张
      │   └── val2017 #验证集图片，约5000张
      ├── labels  #txt标注目录
      │   ├── train2017 #对应目标检测的训练集txt标注文件
      │   └── val2017 #对应目标检测的验证集txt标注文件
      ├── train2017.txt #训练集图片路径
      └── val2017.txt #验证集图片路径

```

### 2.3 构建环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    ruamel.yaml==0.18.6
    git+https://gitee.com/xiwei777/tcap_dllogger.git
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Detection/YOLO12
    ```
    

2. 训练指令。
    - 单机单卡
        ```
        python ./run_scripts/run_yolo12.py \
            --model_name yolo12n \
            --total_epochs 600 \
            --batch_size 32 \
            --device 1 \
            --data_path /data/teco-data/COCO \
            --early_stop 100 \
            --num_workers 8 \
            --autocast True \
            --lr0 0.01 \
            --warmup_bias_lr 0.1 \
            --optimizer auto \
            --imgsz 640 \
            --mosaic 1.0
        ```