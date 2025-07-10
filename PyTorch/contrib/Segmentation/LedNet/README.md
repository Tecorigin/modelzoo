# LEDNet

## 1. 模型概述
LEDNet是一种高效的轻量级深度神经网络，专为实时语义分割任务设计。LEDNet采用了编码-解码架构，其中编码器部分利用可分离卷积和通道剪枝技术，以减少计算量和参数数量，而解码器部分通过逐步恢复空间分辨率，实现高效的特征融合和预测。LEDNet在保持高精度的同时，显著降低了计算复杂度，适合资源受限的环境，如嵌入式设备和移动端应用。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集介绍

我们在本项目中使用了 ISPRS Vaihingen 数据集。该数据集广泛应用于遥感图像分析和建筑物检测任务，提供了高分辨率的航空影像和对应的地面真值标签。这些影像涵盖了不同类型的地物，其类别如下图所示。数据集中的影像经过精确标注，可用于训练和评估深度学习模型。

| 像素值 | 类别名（英文）           | 类别名（中文） |
| ------ | ------------------------- | -------------- | 
| 0      | Other       | 其他     |
| 1      | Impervious surfaces       | 不透光表面     |
| 2      | Building                  | 建筑           |
| 3      | Low vegetation            | 低矮植被       | 
| 4      | Tree                      | 树木           |
| 5      | Car                       | 汽车           |
| 6      | Clutter/background        | 背景           | 


#### 2.2.2 从百度网盘中下载数据集
链接：https://pan.baidu.com/s/1UMbFw0dve9j2fErJ2R4PPQ \
提取码：v2eq

#### 2.2.3 解压数据集

解压训练数据集：

``` bash
unzip vaihingen_VOC.zip -d ./datasets/data
```

#### 2.2.4 数据集目录结构

vaihingen数据集目录结构参考如下所示(按照VOC格式整理):

```
└── vaihingen_VOC
    └── VOC2007
        ├── ImageSets
        │   └── Segmentation
        │       ├── train.txt
        │       └── val.txt
        ├── JPEGImages
        │   ├── area10_0_0_512_512.png
        │   ├── area10_0_1024_512_1536.png
        │   ├── ...
        └── SegmentationClass
            ├── area10_0_0_512_512.png
            ├── area10_0_1024_512_1536.png
            └── ...
```
与VOC2007数据集的结构相同，`ImageSets/Segmentation/`中存放的是数据集划分文件，`JPEGImages/`中存放的是原始影像（三通道png图片），`SegmentationClass/`中存放的是与每张原始影像对应的语义分割标注文件(灰度png图像，像素值代表类别)
train.txt、val.txt 的格式如下：

```
# 每一行均为对应的源图像和标注的文件名前缀

# 下面是 train_list.txt 中的格式样例
area11_0_0_512_512
area11_0_1024_512_1536
...

# 下面是 val_list.txt 中的格式样例
area10_0_0_512_512
area10_0_1024_512_1536
...
```

   > **说明：**
   > 该数据集的训练过程脚本只作为一种参考示例。

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Segmentation/LEDNet

conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"
```
<p align="center">
    <img src="images/env.png" alt="Source Image" width="80%">
</p>

2. 安装python依赖
``` bash
# install requirements
pip install -r requirements.txt
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Segmentation/LEDNet
    ```


注：在vaihingen数据集上训练150epoch，单卡DDP，开启amp得到的pth权重文件可在以下链接中下载，如果需要可以直接下载并将其存放到`/experiments/example`目录中，可以直接进行结果测试和可视化。\
\
链接: https://pan.baidu.com/s/1Wupf1ifRN-K3hAselBv1EQ \
提取码: 4me4 

2. 运行训练。该模型支持单机单SPA、单机单卡（DDP）以及断点重训。训练过程保存的权重以及日志均会保存在--work_dir中。

- 单机单SPA训练
    ```
    python run_scripts/run_lednet.py --device sdaa --work_dir experiments/lednet --data_root ./datasets/data --distributed False --use_amp True --default_rank 0 --model_name lednet --crop_val True --lr 0.005 --optimizer adam --crop_size 512 --batch_size 4 --total_epochs 150 --val_epoch 5
    ```
- 单机单卡训练（DDP）
    ```
    python run_scripts/run_lednet.py --nproc_per_node=4 --master_port=29510 --device sdaa --data_root ./datasets/data --work_dir experiments/lednet --distributed True --use_amp True --default_rank 0 --model_name lednet --crop_val True --lr 0.005 --optimizer adam --crop_size 512 --batch_size 4 --total_epochs 150 --val_epoch 5
    ```

断点重训只需在训练命令中加入--continue_training True, --ckpt {checkpoint file}

更多训练参数参考[README](run_scripts/README.md)

### 2.5 启动测试
- 单机单SPA测试
    ```
    python run_scripts/run_lednet.py --distributed False --device sdaa --data_root ./datasets/data --ckpt experiments/example/best_lednet_vaihingen.pth --use_amp True --test_only True --default_rank 0 --model_name lednet --crop_val True --crop_size 512
    ```
- 单机单卡测试
    ```
    python run_scripts/run_lednet.py --nproc_per_node=4 --master_port=29505 --distributed True --data_root ./datasets/data --device sdaa --ckpt experiments/example/best_lednet_vaihingen.pth --use_amp True --test_only True --default_rank 0 --model_name lednet --crop_val True --crop_size 512
    ```

### 2.6 可视化分割结果
可视化脚本为:`tools/visualizer.py`，其参数列表如下 
参数名 | 解释 | 样例
-----------------|-----------------|-----------------
ckpt | 用于模型推理的checkpoint文件 | --ckpt experiments/example/best_lednet_vaihingen.pth
img_file| 用于输入的图片文件 | --img_file datasets/data/vaihingen_VOC/VOC2007/JPEGImages/area1_0_0_512_512.png
mix | 是否使用mix可视化模式 | --mix True
alpha | mix模式下的分割结果图透明度 | --alpha 0.5
num_classes | 数据集目标类别数，vaihingen为6 | --num_classes 6
model_name | 模型名称 | --model_name lednet
device | device名称 | --device sdaa
- 可视化命令
    ```
    python tools/visualizer.py --device sdaa
    ```
运行可视化脚本后，将会在会checkpoint文件同级目录中创建一个`vis_results/`目录用于存放可视化结果图

### 2.7 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 吞吐量|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| LEDNet |是|4|512*512| 28 img/s|

**训练结果量化指标如下表所示**

| 指标 |值  |
|:-:|:-:|
| Overall Acc | 83.49 |
| Mean Acc | 64.52 |
| FreqW Acc | 71.65 |
| Mean IoU | 63.63 |

| 目标类别 | IoU |
|:-:|:-:|
| Impervious surfaces | 75.34 |
| Building | 77.60 |
| Low vegetation | 61.55 |
| Tree | 74.16 |
| Car | 29.50 |

