# SegNet

## 1. 模型概述

BiSeNet 是一种专为实时语义分割设计的神经网络模型，采用双向最短路径网络结构。它创新性地通过空间路径和上下文路径并行处理信息，并利用特征融合模块进行特征整合，有效解决了实时语义分割中精度与速度难以兼顾的问题。BiSeNet 模型在实时性要求较高的语义分割任务中表现卓越，如自动驾驶场景下的道路场景分割、智能监控中的行人与物体分割等。鉴于其在实时语义分割方面的出色性能和广泛应用，BiSeNet 已成为深度学习和计算机视觉领域处理实时语义分割任务的重要模型之一。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 CamVid 数据集。链接： http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

#### 2.2.2 解压数据集

以下操作已经在data文件夹中完成，数据集可以直接使用

- 解压训练数据集：`LabeledApproved_full.zip`和`camvid.tgz`
- 将`LabeledApproved_full.zip`解压后的文件放在`camvid/labeled`文件夹中
- 将images文件夹中的test.txt删除 或 移动到images文件夹外
#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:

```
camvid
|-- codes.txt
|-- images
|-- labeled
|-- labels
`-- valid.txt
```


### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd PyTorch/contrib/Segmentation/segnet
conda activate segnet
```


### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PyTorch/contrib/Segmentation/segnet
    ```

2. 运行训练。
    ```
  cd ../CamVid
  export SDAA_VISIBLE_DEVICES=0,1,2,3
  torchrun --nproc_per_node 4 Train_SegNet.py 2>&1 | tee ../scripts/train_sdaa_3rd.log 
    ```


### 2.5 训练结果

- 可视化命令
    ```
    cd ./scripts
    python plot_curve.py
    ```
 | 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch | train_loss |
| --- | --- | --- | --- | --- | --- |
| 4 | SegNet | 是 | 2 | 5 | 1.3767 |