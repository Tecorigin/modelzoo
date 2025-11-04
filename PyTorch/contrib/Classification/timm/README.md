# Pytorch Image Models
## 介绍

* Pytorch Image Models（timm）是一个包含图像模型、层、实用工具、优化器、调度器、数据加载器/数据增强以及参考训练/验证脚本的集合，旨在汇集各种 SOTA 模型，并具有复现 ImageNet 训练结果的能力。

## 特征
* 低侵入性：兼容官方 timm 模块，无需修改

* 全流程适配：训练、验证、推理全流程适配

* 模型覆盖全面：支持 ResNet VGG 等 CNN 系列模型，以及 Vit Deit 等 Transformers 系列模型

* 支持预训练模型：兼容各个分类模型的预训练模型，可快速进行微调训练


## 快速指南

### 1、环境准备

#### 1.1 拉取代码仓

``` bash
git clone https://gitee.com/tecorigin/modelzoo.git
```

#### 1.2 Docker 环境准备

##### 1.2.1 获取 SDAA Pytorch 基础 Docker 环境

SDAA 提供了支持 Pytorch 的 Docker 镜像，请参考 [Teco文档中心的教程](http://docs.tecorigin.com/release/tecopytorch/v1.5.0/) -> 安装指南 -> Docker安装 中的内容进行 SDAA Pytorch 基础 Docker 镜像的部署。

##### 1.2.2 激活 Teco Pytorch 虚拟环境
使用如下命令激活并验证 torch_env 环境

``` bash
conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"

--------------+----------------------------------------------
Host IP      | 127.0.0.1
PyTorch      | 2.0.0a0+gitdfe6533
Torch-SDAA   | 1.5.0
--------------+----------------------------------------------
SDAA Driver  | 1.1.1 (N/A)
SDAA Runtime | 1.1.0 (/opt/tecoai/lib64/libsdaart.so)
SDPTI        | 1.1.0 (/opt/tecoai/lib64/libsdpti.so)
TecoDNN      | 1.18.0 (/opt/tecoai/lib64/libtecodnn.so)
TecoBLAS     | 1.18.0 (/opt/tecoai/lib64/libtecoblas.so)
CustomDNN    | 1.18.0 (/opt/tecoai/lib64/libtecodnn_ext.so)
TecoRAND     | 1.5.0 (/opt/tecoai/lib64/libtecorand.so)
TCCL         | 1.15.0 (/opt/tecoai/lib64/libtccl.so)
--------------+----------------------------------------------
```

##### 1.2.3 安装依赖模块
使用如下命令安装依赖模块

``` bash
pip install -r requirements.txt
```

### 2、数据集准备
#### 2.1 获取数据集

ImageNet 1k 是一个来自 ILSVRC 挑战赛的广受欢迎的图像分类数据集。要使用混合精度或 FP32 精度训练您的模型，请根据以下步骤获取并处理数据集：

##### 2.1.1. 从公开网站中获取数据集下载
https://image-net.org/download-images

#### 2.2 解压数据集

解压训练数据集：

``` bash
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```
解压测试数据集：

``` bash
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
cd ..
```
#### 2.3 数据集目录结构

在本文档中，包含 `train/` 和 `val/` 目录被称为 `path to imagenet`，数据集目录结构参考如下所示:

```
└── ImageNet
    ├──train
    │   ├──类别1
    │   │   ├──图片1
    │   │   ├──图片2
    │   │   └── ...
    │   ├──类别2
    │   │   ├──图片1
    │   │   ├──图片2
    │   │   └── ...
    │   └── ...
    ├──val
    │    ├──图片1
    │    ├──图片2
    │    ├──图片3
    │    ├──图片4
    │    └── ...
```

### 3、 启动训练

``` bash
cd PyTorch/contrib/Classification/timm
```
#### 3.1 支持单机单卡、单机四卡、单机八卡、两机八卡

- 运行示例

    下面给出了一个训练 efficientnet_b0 模型的示例脚本，单卡四核组，更多的训练参数可以参考 [run_scripts/README.md](./run_scripts/README.md)。

    ```bash
    python run_scripts/run_train_timm.py --data-dir ./imagenet --model efficientnet_b0 -b 64 --sched step --epochs 20 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.3 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016 --nproc_per_node 4
    ```

        --------------+----------------------------------------------
        Host IP      | 127.0.0.1
        PyTorch      | 2.0.0a0+gitdfe6533
        Torch-SDAA   | 1.5.0
        --------------+----------------------------------------------
        SDAA Driver  | 1.1.1 (N/A)
        SDAA Runtime | 1.1.0 (/opt/tecoai/lib64/libsdaart.so)
        SDPTI        | 1.1.0 (/opt/tecoai/lib64/libsdpti.so)
        TecoDNN      | 1.18.0 (/opt/tecoai/lib64/libtecodnn.so)
        TecoBLAS     | 1.18.0 (/opt/tecoai/lib64/libtecoblas.so)
        CustomDNN    | 1.18.0 (/opt/tecoai/lib64/libtecodnn_ext.so)
        TecoRAND     | 1.5.0 (/opt/tecoai/lib64/libtecorand.so)
        TCCL         | 1.15.0 (/opt/tecoai/lib64/libtccl.so)
        --------------+----------------------------------------------
        Added key: store_based_barrier_key:1 to store for rank: 1
        Added key: store_based_barrier_key:1 to store for rank: 0
        Added key: store_based_barrier_key:1 to store for rank: 2
        Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 3 nodes.
        Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 3 nodes.
        Training in distributed mode with multiple processes, 1 device per process.Process 1, total 3, device sdaa:1.
        Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 3 nodes.
        Training in distributed mode with multiple processes, 1 device per process.Process 0, total 3, device sdaa:0.
        Training in distributed mode with multiple processes, 1 device per process.Process 2, total 3, device sdaa:2.
        Model efficientnet_b0 created, param count:5288548
        Data processing configuration for current model + dataset:
                input_size: (3, 224, 224)
                interpolation: bicubic
                mean: (0.485, 0.456, 0.406)
                std: (0.229, 0.224, 0.225)
                crop_pct: 0.875
                crop_mode: center
        Using native Torch AMP. Training in mixed precision.
        Using native Torch DistributedDataParallel.
        Scheduled epochs: 20. LR stepped per epoch.
        Train: 0 [   0/6672 (  0%)]  Loss: 6.96 (6.96)  Time: 1.839s,  104.41/s  (1.839s,  104.41/s)  LR: 1.000e-04  Data: 0.848 (0.848)
        Train: 0 [  50/6672 (  1%)]  Loss: 6.96 (6.96)  Time: 0.734s,  261.42/s  (0.742s,  258.85/s)  LR: 1.000e-04  Data: 0.010 (0.024)
        Train: 0 [ 100/6672 (  1%)]  Loss: 6.96 (6.96)  Time: 0.704s,  272.54/s  (0.735s,  261.34/s)  LR: 1.000e-04  Data: 0.008 (0.016)
        ...

### 4. 模型精度验证

#### 4.1 模型验证
- 运行示例

    使用下面的命令加载预训练模型进行精度验证，更多模型请参考 [timm 官方 huggingface 仓库](https://huggingface.co/timm)

    ```bash
    # 可通过 HF_ENDPOINT 环境变量修改 huggingface 镜像源
    python validate.py --data-dir ./imagenet --model tf_efficientnet_b0.in1k --pretrained
    ```

        --------------+----------------------------------------------
        Host IP      | 127.0.1.1
        PyTorch      | 2.0.0a0+gitdfe6533
        Torch-SDAA   | 1.5.0
        --------------+----------------------------------------------
        SDAA Driver  | 1.1.1 (N/A)
        SDAA Runtime | 1.1.0 (/opt/tecoai/lib64/libsdaart.so)
        SDPTI        | 1.1.0 (/opt/tecoai/lib64/libsdpti.so)
        TecoDNN      | 1.18.0 (/opt/tecoai/lib64/libtecodnn.so)
        TecoBLAS     | 1.18.0 (/opt/tecoai/lib64/libtecoblas.so)
        CustomDNN    | 1.18.0 (/opt/tecoai/lib64/libtecodnn_ext.so)
        TecoRAND     | 1.5.0 (/opt/tecoai/lib64/libtecorand.so)
        TCCL         | 1.15.0 (/opt/tecoai/lib64/libtccl.so)
        --------------+----------------------------------------------
        Validating in float32. AMP not enabled.
        Loading pretrained weights from Hugging Face hub (timm/tf_efficientnet_b0.in1k)
        [timm/tf_efficientnet_b0.in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
        Model tf_efficientnet_b0.in1k created, param count: 5288548
        Data processing configuration for current model + dataset:
                input_size: (3, 224, 224)
                interpolation: bicubic
                mean: (0.485, 0.456, 0.406)
                std: (0.229, 0.224, 0.225)
                crop_pct: 0.875
                crop_mode: center
        Test: [   0/196]  Time: 4.404s (4.404s,   58.13/s)  Loss:  0.5665 (0.5665)  Acc@1:  89.453 ( 89.453)  Acc@5:  98.047 ( 98.047)
        Test: [  10/196]  Time: 1.639s (1.997s,  128.18/s)  Loss:  1.0224 (0.7711)  Acc@1:  74.219 ( 82.635)  Acc@5:  95.312 ( 95.916)
        Test: [  20/196]  Time: 1.598s (1.850s,  138.38/s)  Loss:  0.7101 (0.7825)  Acc@1:  87.500 ( 82.533)  Acc@5:  93.359 ( 95.536)
        Test: [  30/196]  Time: 1.705s (1.812s,  141.27/s)  Loss:  0.8240 (0.7453)  Acc@1:  80.469 ( 83.506)  Acc@5:  95.312 ( 95.817)
        Test: [  40/196]  Time: 1.880s (1.778s,  144.00/s)  Loss:  0.7671 (0.7788)  Acc@1:  83.203 ( 82.393)  Acc@5:  96.094 ( 95.817)
        Test: [  50/196]  Time: 1.696s (1.745s,  146.69/s)  Loss:  0.5487 (0.7720)  Acc@1:  89.844 ( 82.529)  Acc@5:  98.438 ( 96.071)
        Test: [  60/196]  Time: 1.952s (1.760s,  145.42/s)  Loss:  1.0238 (0.7858)  Acc@1:  75.000 ( 81.999)  Acc@5:  95.312 ( 96.094)
        Test: [  70/196]  Time: 1.653s (1.751s,  146.18/s)  Loss:  0.8080 (0.7696)  Acc@1:  80.859 ( 82.306)  Acc@5:  96.875 ( 96.237)
        Test: [  80/196]  Time: 1.580s (1.757s,  145.70/s)  Loss:  1.4182 (0.7905)  Acc@1:  64.453 ( 81.829)  Acc@5:  89.453 ( 95.915)
        Test: [  90/196]  Time: 1.989s (1.755s,  145.83/s)  Loss:  1.8509 (0.8377)  Acc@1:  52.344 ( 80.606)  Acc@5:  86.328 ( 95.398)
        Test: [ 100/196]  Time: 1.772s (1.744s,  146.79/s)  Loss:  1.3170 (0.8875)  Acc@1:  68.359 ( 79.394)  Acc@5:  89.062 ( 94.767)
        Test: [ 110/196]  Time: 1.899s (1.744s,  146.81/s)  Loss:  0.9266 (0.9093)  Acc@1:  78.906 ( 78.952)  Acc@5:  94.141 ( 94.489)
        Test: [ 120/196]  Time: 1.579s (1.743s,  146.85/s)  Loss:  1.3544 (0.9228)  Acc@1:  68.750 ( 78.790)  Acc@5:  86.719 ( 94.221)
        Test: [ 130/196]  Time: 1.564s (1.738s,  147.30/s)  Loss:  0.7331 (0.9502)  Acc@1:  85.156 ( 78.092)  Acc@5:  96.094 ( 93.914)
        Test: [ 140/196]  Time: 1.840s (1.735s,  147.53/s)  Loss:  1.1114 (0.9638)  Acc@1:  73.828 ( 77.840)  Acc@5:  93.359 ( 93.767)
        Test: [ 150/196]  Time: 1.759s (1.730s,  148.01/s)  Loss:  1.1910 (0.9851)  Acc@1:  75.000 ( 77.395)  Acc@5:  87.109 ( 93.478)
        Test: [ 160/196]  Time: 1.612s (1.728s,  148.12/s)  Loss:  0.9061 (1.0010)  Acc@1:  82.031 ( 77.116)  Acc@5:  94.141 ( 93.248)
        Test: [ 170/196]  Time: 1.585s (1.723s,  148.61/s)  Loss:  0.6834 (1.0179)  Acc@1:  85.156 ( 76.736)  Acc@5:  98.047 ( 93.037)
        Test: [ 180/196]  Time: 1.619s (1.719s,  148.88/s)  Loss:  1.1861 (1.0311)  Acc@1:  69.141 ( 76.368)  Acc@5:  94.922 ( 92.934)
        Test: [ 190/196]  Time: 1.817s (1.718s,  148.97/s)  Loss:  1.0966 (1.0295)  Acc@1:  73.047 ( 76.407)  Acc@5:  95.703 ( 92.973)
        * Acc@1 76.552 (23.448) Acc@5 93.012 (6.988)
        --result
        {
            "model": "tf_efficientnet_b0.in1k",
            "top1": 76.552,
            "top1_err": 23.448,
            "top5": 93.012,
            "top5_err": 6.988,
            "param_count": 5.29,
            "img_size": 224,
            "crop_pct": 0.875,
            "interpolation": "bicubic"
        }

#### 4.2 模型验证列表

* 已测试验证如下模型，精度情况如下：

    |model|top1|top5|
    |:-:|:-:|:-:|
    |tinynet_e.in1k|59.874|81.76|
    |levit_128s.fb_dist_in1k|76.528|92.856|
    |tf_efficientnet_b0.in1k|76.552|93.012|
    |efficientvit_m5.r224_in1k|77.08|93.178|
    |convnextv2_tiny.fcmae_ft_in22k_in1k_384|85.102|97.626|
