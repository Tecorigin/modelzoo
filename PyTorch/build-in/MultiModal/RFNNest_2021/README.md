# RFN-Nest

---

### The re-implementation of Information Fusion 2021 RFN-Nest paper idea

#### framework
![](figure/framework.png)

#### decoder
![](figure/decoder.png)

#### train-rfn
![](figure/training-rfn.png)

This code is based on [Hui Li, Xiao-Jun Wu*, Josef Kittler, "RFN-Nest: An end-to-end residual fusion network for infrared and visible images" in Information Fusion (IF:13.669), Volume: 73, Pages: 72-86, September 2021](https://www.sciencedirect.com/science/article/abs/pii/S1566253521000440?via%3Dihub)

---

## Description 描述

- **基础框架：** AutoEncoder
- **任务场景：** 用于红外可见光图像融合，Infrared Visible Fusion (IVF)。
- **项目描述：** RFN-Nest 的 PyTorch 实现。项目中是先将可见光RGB转为灰度图，进行两个灰度图的融合。
- **论文地址：**
  - [arXiv](https://arxiv.org/abs/2103.04286)
  - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253521000440?via%3Dihub)
- **参考项目：**
  - [imagefusion-rfn-nest](https://github.com/hli1221/imagefusion-rfn-nest) 官方代码。
  - 官方代码的融合策略中还整合了其他一些无需学习的融合算法。用于后续的实验对比。自己的代码里没有写其他融合算法。

---

## Idea 想法

[MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip) (T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.) is utilized to train our AutoEncoder network.

[KAIST](https://soonminhwang.github.io/rgbt-ped-detection/) (S. Hwang, J. Park, N. Kim, Y. Choi, I. So Kweon, Multispectral pedestrian detection: Benchmark dataset and baseline, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1037–1045.) is utilized to train the RFN modules.

对KASIT数据集的介绍可以看：
- https://zhuanlan.zhihu.com/p/722196868
- https://www.cnblogs.com/kongen/p/18080535
---


## Structure 文件结构

```shell
├─ data_test            # 用于测试的不同图片
│  ├─ LLVIP          	# RGB可见光 + Gray红外
│  ├─ Road          	  	# Gray  可见光+红外
│  └─ Tno           		# Gray  可见光+红外
│ 
├─ data_result    # run_infer.py 的运行结果。使用训练好的权重对data_test内图像融合结果 
│ 
├─ models       
│  ├─ fusion_strategy            # 融合策略              
│  └─ NestFuse                   # 网络模型
│ 
├─ runs              # run_train.py 的运行结果
│  ├─ train_autoencoder_byCOCO2014
│  │  ├─ checkpoints # 模型权重
│  │  └─ logs        # 用于存储训练过程中产生的Tensorboard文件
│  ├─ train_rfn_byKAIST
│  ├─ train_autoencoder_byCOCO2014
│  └─ train_rfn_byLLVIP
│
├─ utils      	                # 调用的功能函数
│  ├─ util_dataset.py            # 构建数据集
│  ├─ util_device.py        	    # 运行设备 
│  ├─ util_fusion.py             # 模型推理
│  ├─ util_loss.py            	# 结构误差损失函数
│  ├─ util_train.py            	# 训练用相关函数
│  └─ utils.py                   # 其他功能函数
│ 
├─ configs.py 	    # 模型训练超参数
│ 
├─ run_infer.py   # 该文件使用训练好的权重将test_data内的测试图像进行融合
│ 
└─ run_train.py      # 该文件用于训练模型

```

## Usage 使用说明



### 准备
#### 环境安装
提供了Dockerfile，下面使用Dockerfile安装
1. 构建Dockerfile，
```
docker build -t densefuse:latest ${your_path}/DenseFuse/
```
2. 运行并进入容器
```
docker run -it --name densefuse --net=host -v /mnt/:/mnt -v /mnt_qne00/:/mnt_qne00 --privileged --shm-size=300g densefuse:latest
```
#### 数据集
本次适配使用COCO2017数据集，下载地址：https://cocodataset.org/#home
或者在共享目录/mnt_qne00/dataset/coco/train2017/

### Trainng

#### 从零开始训练

* 参数说明：参考README.md, 需要求改参数可以在test.sh中修改
                                                                           |
* 设置完成参数后，执行**bash run_scripts/test.sh**即可开始训练：
| 参数名           | 说明                                                                              |
|---------------|---------------------------------------------------------------------------------|
| RFN           | 判断训练阶段，if RFN=True，进入第二阶段训练；否则是训练autoencoder                                    |
| image_path_autoencoder    | 用于训练第一阶段的数据集的路径                                                                 |
| image_path_rfn    | 用于训练第二阶段的数据集的路径                                                                 |
| gray          | 为`True`时会进入灰度图训练模式，生成的权重用于对单通道灰度图的融合; 为`False`时会进入彩色RGB图训练模式，生成的权重用于对三通道彩色图的融合; |
| train_num     | `MSCOCO/train2017`数据集包含**118,287**张图像，设置该参数来确定用于训练的图像的数量                        |
| deepsupervision        | 是否使用NestFuse的深度监督训练                                                             |
| resume_nestfuse   | 默认为None，设置为已经训练好的**权重文件路径**时可对该权重进行继续训练，注意选择的权重要与**gray**参数相匹配                  |
| resume_rfn   | 默认为None，设置为已经训练好的**权重文件路径**时可对该权重进行继续训练，是rfn的权重                                 |
| device        | 模型训练设备 cpu or gpu                                                                   |
| batch_size    | 批量大小                                                                                |
| num_workers   | 加载数据集时使用的CPU工作进程数量，为0表示仅使用主进程，（在Win10下建议设为0，否则可能报错。Win11下可以根据你的CPU线程数量进行设置来加速数据集加载） |
| learning_rate | 训练初始学习率                                                                             |
| num_epochs    | 训练轮数                                                                                |

```python
def set_args():
    parser = argparse.ArgumentParser(description="模型参数设置")
    parser.add_argument('--RFN',
                        default=False, type=bool, help='判断训练阶段')
    parser.add_argument('--image_path_autoencoder',
                        default=r'E:/project/Image_Fusion/DATA/COCO/train2017', type=str, help='数据集路径')
    parser.add_argument('--image_path_rfn',
                        default=r'E:/project/Image_Fusion/DATA/RoadScene_dataset', type=str, help='数据集路径')
    parser.add_argument('--gray',
                        default=True, type=bool, help='是否使用灰度模式')
    parser.add_argument('--train_num',
                        default=4, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--deepsupervision', default=False, type=bool, help='是否深层监督多输出')
    parser.add_argument('--resume_nestfuse',
                        default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--resume_rfn',
                        default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size, default=4')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=1e-4, help='select the learning rate, default=1e-2')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
    # 使用parse_args()解析参数
    args = parser.parse_args()
```

* 你可以在运行窗口看到类似的如下信息：

```
autoencoder training by COCO2014:
==================模型超参数==================
----------数据集相关参数----------
image_path_autoencoder: ../dataset/COCO_train2014
image_path_rfn: None
gray_images: True
train_num: 80000
----------训练相关参数----------
RFN: False
deepsupervision: False
resume_nestfuse: None
resume_rfn: None
device: cuda
batch_size: 4
num_workers: 0
num_epochs: 4
learning rate : 0.0001
==================模型超参数==================
设备就绪...
Tensorboard 构建完成，进入路径：./runs\train_01-24_10-43\logs_Gray_epoch=4
然后使用该指令查看训练过程：tensorboard --logdir=./
Loaded 80000 images
----autoencoder---- 阶段训练数据载入完成...
测试数据载入完成...
initialize network with normal type
网络模型及优化器构建完成...
Epoch [1/4]: 100%|██████████| 20000/20000 [1:37:00<00:00,  3.44it/s, pixel_loss=0.0000, ssim_loss=0.0000, lr=0.000100]
Epoch [2/4]: 100%|██████████| 20000/20000 [1:30:42<00:00,  3.67it/s, pixel_loss=0.0000, ssim_loss=0.0000, lr=0.000090]
Epoch [3/4]: 100%|██████████| 20000/20000 [1:31:32<00:00,  3.64it/s, pixel_loss=0.0000, ssim_loss=0.0000, lr=0.000081]
Epoch [4/4]: 100%|██████████| 20000/20000 [1:33:18<00:00,  3.57it/s, pixel_loss=0.0000, ssim_loss=0.0000, lr=0.000073]
Finished Training
训练耗时：22354.39秒
Best loss: 0.002613












