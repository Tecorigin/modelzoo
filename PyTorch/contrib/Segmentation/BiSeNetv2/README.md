# BiSeNetV2

## 1. 模型概述

BiSeNet 是一种专为实时语义分割设计的神经网络模型，采用双向最短路径网络结构。它创新性地通过空间路径和上下文路径并行处理信息，并利用特征融合模块进行特征整合，有效解决了实时语义分割中精度与速度难以兼顾的问题。BiSeNet 模型在实时性要求较高的语义分割任务中表现卓越，如自动驾驶场景下的道路场景分割、智能监控中的行人与物体分割等。鉴于其在实时语义分割方面的出色性能和广泛应用，BiSeNet 已成为深度学习和计算机视觉领域处理实时语义分割任务的重要模型之一。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 Cityscapes 数据集。链接：https://www.cityscapes-dataset.com/

#### 2.2.2 解压数据集

- 解压训练数据集：

``` bash
cd /data/datasets/20241122/Cityscapes/
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
```


- 将`./data/cityscapes/`目录下的`train.txt`和`val.txt`复制到`/data/datasets/20241122/Cityscapes/`中 
#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:

```
Cityscapes/
|-- gtFine
|   |-- test
|   |-- train
|   `-- val
`-- leftImg8bit
    |-- test
    |-- train
    `-- val
```


### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd PyTorch/contrib/Segmentation/BiSeNetv2
conda activate BiSeNet
```
2. 安装python依赖
``` bash
pip install -r requirements.txt
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PyTorch/contrib/Segmentation/BiSeNetv2
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=2,3
    cfg_file=configs/bisenetv2_city.py
    python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 6666 tools/train_amp.py --config $cfg_file
    ```


### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
 只能训练40iter，无法判断loss趋势