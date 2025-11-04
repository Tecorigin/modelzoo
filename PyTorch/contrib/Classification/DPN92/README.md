# DPN92

## 1. 模型概述
DPN92（Deep Pyramid Networks）是深度卷积神经网络（CNN）的一种架构，旨在通过有效利用多尺度特征来提高图像分类和目标检测等任务的性能。DPN的核心思想是将不同的卷积层特征进行融合，形成一种“金字塔”结构，以便捕捉不同尺度的特征信息。这种方法能够提升模型对不同大小物体的感知能力。

源码链接: https://github.com/rwightman/pytorch-dpn-pretrained

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

#### 2.2.1 数据集位置
/data/datasets/imagenet


### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/DPN92

conda activate torch_env

2. 安装python依赖
``` 
bash
# install requirements
pip install -r requirements.txt
```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/DPN92
    ```

2.每个iteration的loss图会保存在/output路径中
- 单机单核组
    ```
    python validate.py /data/datasets/imagenet/train
    ```
- 单机单卡
    ```
    python validate.py /data/datasets/imagenet/train --multi-gpu --dist-url 'tcp://127.0.0.1:65501' --world-size 1 --rank 0 --dist-backend 'tccl' --gpu 0

更多训练参数参考[README](run_scripts/README.md)


### 2.5 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| DPN92 |是|4|300*300|

**训练结果量化指标如下表所示**

| 训练数据集 | 输入图片大小 |nproc_per_node|accuracy(DDP)|
| :-----: | :-----: |:------: |:------: |
| mini_imagenet | 224x224 |4|99.9% |





