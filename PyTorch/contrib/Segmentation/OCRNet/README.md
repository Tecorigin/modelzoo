# OCRNet

## 1. 模型概述

OCRNet即Object-Contextual Representations Network，是一种专为语义分割任务设计的深度神经网络模型。它的核心在于提出了对象上下文表示（Object-Contextual Representations）的概念，通过聚合全局范围内与目标对象相关的上下文信息，对每个像素的特征进行增强。具体而言，OCRNet利用注意力机制，计算目标像素与图像中其他像素之间的相关性，从而挖掘出更具判别性的特征，有效解决了语义分割中物体边界模糊和上下文信息利用不足的问题。在结构上，它通常基于经典的骨干网络（如ResNet等），并结合独特的对象上下文模块，在复杂场景的语义分割任务中表现出色，如自然场景图像分割、遥感图像分析等。凭借其强大的上下文建模能力和对细节的精准把握，OCRNet已成为深度学习语义分割领域的重要模型之一，为实现高精度的语义分割提供了有效方案，推动了计算机视觉相关应用的发展。 。

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
    cd PyTorch/contrib/Segmentation/OCRNet
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    sh tools/dist_train.sh  ./configs/ocrnet/ocrnet_hr18_4xb2-40k_cityscapes-512x1024.py 4 | tee ./scripts/train_sdaa_3rd.log
    ```

### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
 2h训练了3700个iter
| 加速卡数量 | 模型 | 混合精度 | Batch Size | iter | train_loss |
| --- | --- | --- | --- | --- | --- | 
| 4 | OCRNet | 是 | 4 | 1435 | 0.6168 |