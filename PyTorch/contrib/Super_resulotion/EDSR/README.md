# EDSR

## 1.模型概述

EDSR是一种基于深度学习的图像超分辨率模型，通过增强的残差网络结构显著提升图像的分辨率和质量。来自于[mmagic](https://github.com/open-mmlab/mmagic)，使用DIV2K数据进行训练，训练环境为`mmagic`。

## 2.快速开始

### 2.1环境配置

请参考[基础环境安装](https://gitee.com/tecorigin/modelzoo/blob/main/doc/Environment.md)章节，完成训练前的基础环境检查和安装。

```bash
pip install -r requirements.txt
pip install -e .
```
并且需要mmengine以及mmcv。

### 2.2数据集获取与准备

数据集来自[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)，运行以下命令进行数据预处理。
```bash
python tools/dataset_converters/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

### 2.3模型训练

运行以下命令进行训练。
```bash
bash configs/edsr/train.sh
```

### 2.4结果储存

训练过程中的loss曲线如下图所示：

![loss](loss.png)

可以看出随着训练进行，loss不断下降。