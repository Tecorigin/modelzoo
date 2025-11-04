# Pix2Pix

## 1. 模型概述

pix2pix模型是一种基于条件生成对抗网络（Conditional GAN）的图像到图像翻译模型，它通过引入对抗训练和条件约束，实现了从一个领域到另一个领域图像风格的转换。pix2pix模型的生成器将输入图像与条件信息相结合，生成与目标领域风格相似的图像。这种模型结构不仅提高了图像转换的质量，还保证了转换过程的连续性和真实性。  


## 2. 快速开始

使用本模型执行训练的主要流程如下：  
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。  
2. 获取数据集：介绍如何获取训练所需的数据集。  
3. 启动训练：介绍如何运行训练。  

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

另外，训练代码基于源代码百度飞桨生成对抗网络开发套件PaddleGAN`ppgan`库修改，为适配sdaa应修改`/ppgan/utils/setup.py`第47-54行，添加
```python
-45- paddle.set_device("sdaa")
```
![edit](https://foruda.gitee.com/images/1737798779258562475/ee29f4d4_15340282.png "edit.png")

复现环境设置`paddlepaddle==2.6.0`，`numpy==1.26.4`，`paddle-sdaa==2.0.0`，建议安装适配的`opencv-python`如`4.11.0.86`版本。

### 2.2 准备数据集

Pix2Pix运行在Facades数据集上，数据集配置可以参考 https://aistudio.baidu.com/datasetdetail/230639

Facades正确解压后文件结构如下

```bash
facades/
├── train/ # 目录包含了用于训练的图像对
│   ├── input/ # 子目录包含了原始的图像
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   ├── ...
│   ├── target/ # 子目录包含了对应的期望输出图像
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   ├── ...
├── test/ # 目录包含了用于测试的图像对
│   ├── input/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   ├── ...
│   ├── target/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   ├── ...
└── val/
    ├── input/
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   ├── 3.jpg
    │   ├── ...
    ├── target/
        ├── 1.jpg
        ├── 2.jpg
        ├── 3.jpg
        ├── ...
```

### 2.3 启动训练

该模型参考源码仅支持单机单核组  

**配置Python环境**

```bash
pip install -r requirements.txt

cd <modelzoo-dir>PaddlePaddle/contrib/GAN/Pix2Pix/scripts
```

**单机单核组**

```bash
python -u tools/main.py --config-file configs/pix2pix_facades.yaml
```

### 2.4 训练结果
模型训练50轮Epoch训练约1h，训练结果如下  

| 加速卡数量 |     模型      | 混合精度 | Batch Size | Epoch | D_fake_loss | G_L1_loss |
| :--------: | :-----------: | :------: | :--------: | :---: | :--------: | :-----: |
|     1      | Pix2Pix |   是    |     1     |   50   |   0.705   |  99.86  |