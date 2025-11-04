# PyTorch Image Models
timm（Pytorch Image Models）项目是一个站在大佬肩上的图像分类模型库，通过timm可以轻松的搭建出各种sota模型（目前内置预训练模型592个，包含densenet系列、efficientnet系列、resnet系列、vit系列、vgg系列、inception系列、mobilenet系列、xcit系列等等），并进行迁移学习。下面对timm的两个模型SEResNet34和efficientnet_b2迁移到加速卡上进行训练。此外，timm的作者在官网实现了timm内置模型的train、validate、inference，在这里不做累述与转载，更多内容请参考官网。

## 环境配置：
- Python3.10
- torch 2.4.0
- torch_sdaa 2.0.0
- torchvision 0.15.1
- Ubuntu或Centos
- 建议使用GPU训练
- 详细环境配置见`requirements.txt`

## 使用说明

### Trainng

#### 快速开始
#### 数据集准备
- ImageNet数据集官方下载地址：https://www.image-net.org/

#### 起docker环境
使用该镜像
```
docker pull jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0
```
创建环境
```
docker run -itd --name=<name> --net=host -v /data/application/hongzg:/data/application/hongzg -v /mnt/:/mnt -v /hpe_share/:/hpe_share -p 22 -p 8080 -p 8888 --privileged --device=/dev/tcaicard20 --device=/dev/tcaicard21 --device=/dev/tcaicard22 --device=/dev/tcaicard23 --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 300g jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0 /bin/bash
```
其他依赖库 参考requirements.txt

#### 训练指令：
SEResNet34
```
./distributed_train.sh 1 /mnt/nvme1/application/hongzg/dataset/timm_test --model seresnet34 --sched cosine --epochs 3 --warmup-epochs 5 --lr 0.001 --reprob 0.5 --remode pixel --batch-size 8 --amp --amp-dtype float16
```
efficientnet_b2
```
./distributed_train.sh 1 /mnt/nvme1/application/hongzg/dataset/timm_test --model efficientnet_b2 -b 16 --sched step --epochs 2 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --amp-dtype float16 --lr 0.01
```
