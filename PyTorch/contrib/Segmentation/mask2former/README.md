# Mask2Former

## 1. 模型概述

Mask2Former 即 Masked-Attention Mask Transformer，是一种用于图像分割任务的先进神经网络模型。它创新性地将 Transformer 架构与分割任务相结合，通过掩码注意力机制（Masked Attention）对图像中的不同区域进行细致的特征提取和处理。模型能够同时处理语义分割、实例分割和全景分割等多种分割任务，通过统一的架构和训练方式，有效避免了传统方法中不同分割任务需要不同模型的局限性。在训练过程中，Mask2Former 通过学习对图像中的不同物体和区域进行掩码生成和分类，从而实现高精度的分割结果。在计算机视觉领域，无论是复杂场景下的目标实例分割，还是对整个场景的语义理解和分割，Mask2Former 都展现出了强大的性能和良好的泛化能力，凭借其多任务处理能力和先进的架构设计，成为了图像分割领域的重要模型之一，为相关研究和应用提供了新的思路和解决方案。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 Cityscapes 数据集。链接：https://www.cityscapes-dataset.com/

#### 2.2.2 处理数据集

- 解压训练数据集：
- 现在/data/datasets/cityscapes/中包含已经转换好的数据集，因此可以直接使用，无需运行下述命令。
``` bash
cd /data/datasets/cityscapes
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
python -m pip install cityscapesscripts
python tools/dataset_converters/cityscapes.py /data/datasets/cityscapes/  
```

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
1. 执行以下命令，启动环境。
参考[官方教程](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/get_started.md)进行安装
```sh
conda create -n mmseg--clone torch_env
conda activate mmseg
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
- 若在安装mmengine时报错："GLIBCXX not found"
参考以下[博客](https://zhuanlan.zhihu.com/p/685165815)和代码解决：
```sh
cd /root/anaconda3/envs/mmseg/lib/ 
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30  .
ln -s -f libstdc++.so.6.0.30 libstdc++.so.6
```
- 注意： mmseg库需要修改site-package中的mmengine环境依赖。建议直接clone迁移好的mmseg环境。

2. 下载代码
``` 
git clone https://github.geekery.cn/https://github.com/open-mmlab/mmsegmentation.git
```
3. 安装依赖
```
pip install -v -e .
pip install mmcv==2.1.0
pip install ftfy
pip install regex
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PyTorch/contrib/Segmentation/mask2former
    ```

2. 运行训练。
    ```
    cd ..
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    sh tools/dist_train.sh ./configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py 4 | tee ./scripts/train_sdaa_3rd.log
    ```

### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
2h训练了395个iter

| 加速卡数量 | 模型 | 混合精度 | Batch Size | iter | train_loss |
| --- | --- | --- | --- | --- | --- | 
| 4 | mask2former | 是 | 16 |395 | 59.1163 |