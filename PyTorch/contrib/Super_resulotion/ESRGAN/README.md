# ESRGAN

## 1.模型概述

ESRGAN是超分辨率生成对抗网络（SRGAN）的增强版，用于从低分辨率图像生成高分辨率图像，重点改善感知质量。来自于[mmagic](https://github.com/open-mmlab/mmagic)，使用DIV2K数据进行训练，训练环境为`mmagic`。

## 2.快速开始

### 2.1环境配置

请参考[基础环境安装](https://gitee.com/tecorigin/modelzoo/blob/main/doc/Environment.md)章节，完成训练前的基础环境检查和安装。

```bash
pip install -r requirements.txt
pip install -e .
```

需要安装mmengine以及mmcv。

### 2.2数据集获取与准备

数据集来自[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)，储存在`./data/DIV2K`中。运行以下命令进行数据预处理。
```bash
python tools/dataset_converters/div2k/preprocess_div2k_dataset.py --data-root ./data/DIV2K
```

### 2.3模型训练

运行以下命令进行训练。
```bash
bash configs/esrgan/train.sh
```

### 2.4结果展示

训练过程中的perceptual loss如下图所示。

![loss figure](loss.png)

可以看出随着训练的进行，perceptual loss呈现下降趋势。
