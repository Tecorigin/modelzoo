## PSPnet：Pyramid Scene Parsing Network语义分割模型在Pytorch当中的实现
---
PSPNet（Pyramid Scene Parsing Network）是一种用于语义分割的深度学习模型。它通过引入金字塔池化模块（PPM），能够从不同的尺度提取图像的上下文信息，从而提高了图像分割的精度。

快速开始 使用本模型执行训练的主要流程如下：
基础环境安装：介绍训练前需要完成的基础环境检查和安装。 获取数据集：介绍如何获取训练所需的数据集。 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。 启动训练：介绍如何运行训练。 
2.1 基础环境安装 请参考基础环境安装章节，完成训练前的基础环境检查和安装。
2.2 准备数据集 
2.2.1 获取数据集
/mnt/nvme1/dnn/houjx/project/segformer-pytorch-master/VOCdevkit
2.3 构建Docker环境 使用Dockerfile，创建运行模型训练所需的Docker环境。
例如：
docker pull jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0   jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.0b0-torch_sdaa2.1.0b0

docker run -itd --name={name} --net=host -v /mnt/nvme1/dnn/houjx:/mnt/nvme1/dnn/houjx -v /mnt/:/mnt -v /hpe_share/:/hpe_share -p 22 -p 8080 -p 8888 --device=/dev/tcaicard0 --device=/dev/tcaicard1 --device=/dev/tcaicard2 --device=/dev/tcaicard3 --device=/dev/tcaicard4 --device=/dev/tcaicard5 --device=/dev/tcaicard6 --device=/dev/tcaicard7 --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 300g jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0 /bin/bash

docker exec -it {name} bash


### 所需环境
torch==1.2.0  

### 文件下载
训练所需的pspnet_mobilenetv2.pth和pspnet_resnet50.pth可在百度网盘中下载。    
链接: https://pan.baidu.com/s/1Ecz-l6lFcf6HmeX_pLCXZw 提取码: wps9    

VOC拓展数据集的百度网盘如下：  
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng 提取码: 44mk   

### 训练步骤
#### a、训练voc数据集
1、将我提供的voc数据集放入VOCdevkit中（无需运行voc_annotation.py）。  
2、在train.py中设置对应参数，默认参数已经对应voc数据集所需要的参数了，所以只要修改backbone和model_path即可。  
3、运行train.py进行训练。  
python train.py