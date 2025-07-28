# RetinaNet
## 1. 模型概述
RetinaNet是一种基于单阶段(one-stage)架构的目标检测模型，由Facebook AI Research团队在2017年提出，其核心创新是通过Focal Loss解决目标检测中正负样本极度不平衡的问题‌12。该模型结合了ResNet骨干网络和特征金字塔(FPN)，实现了多尺度特征融合，显著提升了检测精度‌。

- 论文链接：[[1708.02002]] Focal Loss for Dense Object Detection(https://arxiv.org/abs/1708.02002)
- 仓库链接：https://github.com/open-mmlab/mmdetection/tree/main/configs/retinanet

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
本项目使用 COCO 检测数据集，可从官方 [COCO官网](https://cocodataset.org/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://cloud.tencent.com/developer/article/2420174。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip3 install  -U openmim 
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip3 install opencv_python mmcv --no-deps
    mim install -e .
    pip install -r requirements.txt

    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
  ```
  cd <ModelZoo_path>/PyTorch/contrib/Detection/retinanet/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_retinanet.py --config ../configs/retinanet/retinanet_r50_fpn_1x_coco.py\
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4  \
    --master_port=29500  --amp   --launcher pytorch \
    --cfg-options \
      "train_dataloader.dataset.data_root=$data_path" \
      "val_dataloader.dataset.data_root=$data_path" \
      "val_evaluator.ann_file=$data_path/annotations/instances_val2017.json" 2>&1 | tee sdaa.log
  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
### 2.5 训练结果
|模型             |    数据集      |    sdaa结果       |   cuda结果         |  sdaa耗时    |    
|:---------------|:--------------:|:-----------------:|:-----------------:|:-------------:|
|retinanet_r50    |coco           |0.364               |  0.367           |41h

bbox_mAP_copypaste: 0.364 0.556 0.387 0.210 0.402 0.476 
coco/bbox_mAP: 0.3640  
coco/bbox_mAP_50: 0.5560  
coco/bbox_mAP_75: 0.3870  
coco/bbox_mAP_s: 0.2100  
coco/bbox_mAP_m: 0.4020  
coco/bbox_mAP_l: 0.4760  
data_time: 0.0082  time: 0.1927v