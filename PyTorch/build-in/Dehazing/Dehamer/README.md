# Image Dehazing Transformer with Transmission-Aware 3D Position Embedding (CVPR2022)

This repository contains the official implementation of the following paper:
> **Image Dehazing Transformer with Transmission-Aware 3D Position Embedding**<br>
> Chun-Le Guo, Qixin Yan, Saeed Anwar, Runmin Cong, Wenqi Ren, Chongyi Li<sup>*</sup><br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2022<br>
**Paper Link:** [[official link](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Image_Dehazing_Transformer_With_Transmission-Aware_3D_Position_Embedding_CVPR_2022_paper.pdf)] 

## Overview
![overall_structure](./figs/pipeline.jpg)
Overview structure of our method. Our method consists of five key modules: a transmission-aware 3D position embedding module, a Transformer module, a CNN encoder module, a feature modulation module, and a CNN decoder module.
## 准备
### 数据集
 本次适配采用的训练数据是NH-HAZE去雾数据集
 下载地址：
 [Baidu Cloud](https://pan.baidu.com/s/1RGaVJ5kbd-cokE8ZAF_THw?pwd=801y)
 [Google drive](https://drive.google.com/file/d/1qPYGkCfVgn1Ami7ksf0DmKeKsoHVnm8i/view?usp=sharing)
数据集格式如下：
```
 |-NH-Haze
      |- train_NH
         |- haze
            |- 01_hazy.png 
            |- 02_hazy.png
         |- clear_images
            |- 01_GT.png 
            |- 02_GT.png
         |- trainlist.txt
      |- valid_NH
         |- input 
            |- 51_hazy.png 
            |- 52_hazy.png
         |- gt
            |- 51_GT.png 
            |- 52_GT.png
         |- val_list.txt
```
### 安装环境
提供了Dockerfile，下面使用Dockerfile安装
1. 构建Dockerfile，
```
docker build -t dehamer:latest ${your_path}/Dehamer/
```
2. 运行并进入容器
```
docker run -it --name dehamer --net=host -v /mnt/:/mnt -v /mnt_qne00/:/mnt_qne00 --privileged --shm-size=300g dehamer:latest
```
## 训练
使用run_scripts目录下test.sh脚本一键安装环境训练，具体训练参数可以在脚本中查看
```
bash test.sh
```
训练输出：
```
Train time: 0:00:08 | Valid time: 0:00:05 | Valid loss: 0.14306 | Avg PSNR: 14.86 dB
Saving checkpoint to: ./ckpts/NH/dehamer-NH.pt

EPOCH 22 / 50
Batch  1 [=>                      ] Train loss: 0.00000
Batch  2 [=>                      ] Train loss: 0.14641                                                                            
Batch  2 / 50 | Avg loss: 0.19057 | Avg train time / batch: 74 ms
Batch  3 [=>                      ] Train loss: 0.00000                                                                             
Batch  3 / 50 | Avg loss: 0.09584 | Avg train time / batch: 56 ms
Batch  4 [=>                      ] Train loss: 0.00000                                                                             
Batch  4 / 50 | Avg loss: 0.06765 | Avg train time / batch: 64 ms
Batch  5 [=>                      ] Train loss: 0.00000                                                                               
Batch  5 / 50 | Avg loss: 0.13795 | Avg train time / batch: 66 ms
Batch  6 [=>                      ] Train loss: 0.00000                                                                               
Batch  6 / 50 | Avg loss: 0.11026 | Avg train time / batch: 59 ms
Batch  7 [=>                      ] Train loss: 0.00000                                                                            
Batch  7 / 50 | Avg loss: 0.14676 | Avg train time / batch: 59 ms
Batch  8 [=>                      ] Train loss: 0.00000                                                                               
Batch  8 / 50 | Avg loss: 0.17055 | Avg train time / batch: 67 ms
Batch  9 [=>                      ] Train loss: 0.00000                                                                              
Batch  9 / 50 | Avg loss: 0.06595 | Avg train time / batch: 58 ms
Batch 10 [=>                      ] Train loss: 0.00000                                                                              
Batch 10 / 50 | Avg loss: 0.14240 | Avg train time / batch: 66 ms
Batch 11 [=>                      ] Train loss: 0.00000                                                                            
Batch 11 / 50 | Avg loss: 0.24396 | Avg train time / batch: 70 ms
Batch 12 [=>                      ] Train loss: 0.00000                                                                               
Batch 12 / 50 | Avg loss: 0.06341 | Avg train time / batch: 76 ms
Batch 13 [=>                      ] Train loss: 0.00000                                                                                
Batch 13 / 50 | Avg loss: 0.10660 | Avg train time / batch: 70 ms
Batch 14 [=>                      ] Train loss: 0.00000                                                                               
Batch 14 / 50 | Avg loss: 0.16737 | Avg train time / batch: 64 ms
Batch 15 [=>                      ] Train loss: 0.00000                                                                                
Batch 15 / 50 | Avg loss: 0.06316 | Avg train time / batch: 63 ms
Batch 16 [=>                      ] Train loss: 0.00000                                                                                
Batch 16 / 50 | Avg loss: 0.11583 | Avg train time / batch: 56 ms
Batch 17 [=>                      ] Train loss: 0.00000                                                                               
Batch 17 / 50 | Avg loss: 0.09643 | Avg train time / batch: 57 ms
Batch 18 [=>                      ] Train loss: 0.00000                                                                               
Batch 18 / 50 | Avg loss: 0.08434 | Avg train time / batch: 58 ms
Batch 19 [=>                      ] Train loss: 0.00000                                                                                
Batch 19 / 50 | Avg loss: 0.21684 | Avg train time / batch: 63 ms
```
## 训练结果
![image](https://github.com/user-attachments/assets/767d859e-2d60-443d-8cbe-1fa9459c7488)
