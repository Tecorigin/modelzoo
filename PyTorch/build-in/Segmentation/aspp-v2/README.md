# ASPP-V2(ASPP in DeepLabV2)

## 论文地址
* https://arxiv.org/abs/1606.00915

## 该项目主要参考下面链接, 进行微小改动
* https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation
* https://github.com/kazuto1011/deeplab-pytorch

## 环境配置：
![image](https://github.com/user-attachments/assets/ea7a194d-a8ce-4c27-9d45-f458eb92a6bc)


## 文件结构：
```
  ├── src: 模型的backbone以及LRASPP的搭建
  ├── run_scrpts: 模型的一键运行脚本以及loss对比
  ├── train_utils: 训练、验证以及多GPU训练相关模块
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train.py: 单GPU训练脚本
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的mIoU等指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```
 
## 数据集，本例程使用的是PASCAL VOC2012数据集
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

## 训练方法
* 环境安装
  提供了Dockerfile，下面使用Dockerfile安装
1. 构建Dockerfile，
```
docker build -t densefuse:latest ${your_path}/aspp-v2/
```
2. 运行并进入容器
```
docker run -it --name aspp-v2 --net=host -v /mnt/:/mnt -v /mnt_qne00/:/mnt_qne00 --privileged --shm-size=300g aspp-v2:latest
```
* 确保提前准备好数据集
* 若要使用单GPU或者CPU训练，直接使用run_scrpts/test.sh一键训练
* 若要使用多GPU训练，使用```torchrun --nproc_per_node=8 train_multi_GPU.py```指令,```nproc_per_node```参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上```CUDA_VISIBLE_DEVICES=0,3```(例如我只要使用设备中的第1块和第4块GPU设备)
* ```CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py```

## 注意事项
* 在使用训练脚本时，注意要将'--data-path'(VOC_root)设置为自己存放'VOCdevkit'文件夹所在的**根目录**
* 在使用预测脚本时，要将'weights_path'设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改'--num-classes'、'--data-path'和'--weights'即可，其他代码尽量不要改动

## 训练结果
```
--------------+----------------------------------------------
 Host IP      | 127.0.1.1
 PyTorch      | 2.4.0a0+git4451b0e
 Torch-SDAA   | 2.0.0
--------------+----------------------------------------------
 SDAA Driver  | 2.1.1 (N/A)
 SDAA Runtime | 2.0.0 (/opt/tecoai/lib64/libsdaart.so)
 SDPTI        | 1.3.1 (/opt/tecoai/lib64/libsdpti.so)
 TecoDNN      | 2.0.0 (/opt/tecoai/lib64/libtecodnn.so)
 TecoBLAS     | 2.0.0 (/opt/tecoai/lib64/libtecoblas.so)
 CustomDNN    | 1.22.0 (/opt/tecoai/lib64/libtecodnn_ext.so)
 TecoRAND     | 1.8.0 (/opt/tecoai/lib64/libtecorand.so)
 TCCL         | 1.21.0 (/opt/tecoai/lib64/libtccl.so)
--------------+----------------------------------------------
Epoch: [0]  [  0/366]  eta: 0:03:03  lr: 0.000000  loss: 3.2891 (3.2891)  time: 0.5015  data: 0.2680  max mem: 693
Epoch: [0]  [ 10/366]  eta: 0:01:29  lr: 0.000003  loss: 3.2266 (3.2809)  time: 0.2516  data: 0.0245  max mem: 720
Epoch: [0]  [ 20/366]  eta: 0:01:23  lr: 0.000006  loss: 3.2461 (3.2613)  time: 0.2276  data: 0.0001  max mem: 720
Epoch: [0]  [ 30/366]  eta: 0:01:19  lr: 0.000009  loss: 3.2305 (3.2615)  time: 0.2303  data: 0.0001  max mem: 720
Epoch: [0]  [ 40/366]  eta: 0:01:16  lr: 0.000011  loss: 3.2031 (3.2540)  time: 0.2303  data: 0.0001  max mem: 720
Epoch: [0]  [ 50/366]  eta: 0:01:14  lr: 0.000014  loss: 3.1992 (3.2442)  time: 0.2306  data: 0.0001  max mem: 720
Epoch: [0]  [ 60/366]  eta: 0:01:11  lr: 0.000017  loss: 3.2422 (3.2351)  time: 0.2320  data: 0.0001  max mem: 720
Epoch: [0]  [ 70/366]  eta: 0:01:09  lr: 0.000019  loss: 3.1133 (3.2338)  time: 0.2328  data: 0.0001  max mem: 720
Epoch: [0]  [ 80/366]  eta: 0:01:07  lr: 0.000022  loss: 3.1602 (3.2296)  time: 0.2367  data: 0.0001  max mem: 720
Epoch: [0]  [ 90/366]  eta: 0:01:05  lr: 0.000025  loss: 3.1191 (3.2253)  time: 0.2486  data: 0.0001  max mem: 720
Epoch: [0]  [100/366]  eta: 0:01:03  lr: 0.000028  loss: 3.2422 (3.2237)  time: 0.2517  data: 0.0001  max mem: 720
Epoch: [0]  [110/366]  eta: 0:01:00  lr: 0.000030  loss: 3.1602 (3.2216)  time: 0.2413  data: 0.0001  max mem: 720
Epoch: [0]  [120/366]  eta: 0:00:58  lr: 0.000033  loss: 3.0273 (3.2136)  time: 0.2418  data: 0.0001  max mem: 720
Epoch: [0]  [130/366]  eta: 0:00:56  lr: 0.000036  loss: 3.2070 (3.2101)  time: 0.2454  data: 0.0001  max mem: 720
Epoch: [0]  [140/366]  eta: 0:00:54  lr: 0.000039  loss: 3.3027 (3.2067)  time: 0.2408  data: 0.0001  max mem: 720
Epoch: [0]  [150/366]  eta: 0:00:51  lr: 0.000041  loss: 3.0703 (3.2015)  time: 0.2382  data: 0.0001  max mem: 720
Epoch: [0]  [160/366]  eta: 0:00:49  lr: 0.000044  loss: 3.2148 (3.1983)  time: 0.2420  data: 0.0001  max mem: 720
Epoch: [0]  [170/366]  eta: 0:00:46  lr: 0.000047  loss: 3.0801 (3.1929)  time: 0.2394  data: 0.0001  max mem: 720
```
![image](https://github.com/user-attachments/assets/d6506aa6-2dd3-4eac-a2f8-d31aca8b2f2d)
