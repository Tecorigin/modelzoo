# U2Fusion
* 基于https://github.com/Cat2eacher/Image-Fusion/blob/main/U2Fusion_2020 略有改动


 **U2Fusion是一种统一的无监督图像融合网络。以下是关于U2Fusion的介绍：** 

 **研究背景** 

传统的图像融合方法受限于融合规则选择的有限性和人工设计的复杂性，性能提升受限。端到端模型通常依赖监督学习的地面实况或专门设计的无监督学习度量，但多任务通用的地面实况或无参考度量不存在。不同融合任务中，待整合的重要信息随源图像类型变化很大，限制了大多数方法的通用性。

 **核心思想** 

利用神经网络强大的特征表示能力，统一表示多种信息，形成统一的融合框架。通过特征提取和信息度量，自动估计源图像的重要性，提出自适应信息保存度。训练网络以保持融合结果与源图像之间的自适应相似性，无需地面实况。
网络结构
特征提取：使用预训练的VGG-16模型，提取浅层和深层特征，综合表示基本信息。
信息度量：采用图像梯度作为度量指标，评估特征中信息的重要程度，分配融合权重。
融合网络：使用DenseNet模块生成融合结果，由10个层组成，包含卷积和激活函数，前七层采用密集连接块。

* 论文地址：
  - [IEEE](https://arxiv.org/abs/2007.00328)

## 文件结构
```shell
├─fusion_test_data
│  ├─Road
│  └─Tno
│ 
├─fusion_result
│ 
├─models                  
│  └─DenseNet.py 
│ 
├─runs
│  └─train_04-02_14-43
│     ├─checkpoints
│     └─logs
|
├─utils      	                # 调用的功能函数
│  ├─util_dataset.py            # 构建数据集
│  ├─util_device.py        	# 运行设备 
│  ├─util_fusion.py             # 模型推理
│  ├─util_loss.py            	# 结构误差损失函数
│  ├─util_train.py            	# 训练用相关函数
│  ├─utils.py            	# 其他功能函数
│  └─vgg.py                   # VGG特征提取
│ 
├─scripts
│  ├─run_train.py
│  ├─run_fusion.py
│  ├─config.py

```


## 使用说明

### Trainng

#### 快速开始
* 获取数据集
[官方下载地址](https://cocodataset.org/)
```
存放路径： /mnt/nvme1/dataset/datasets/coco2017
```
* 起docker环境

使用该镜像
```
docker pull jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0
```
创建环境
```
docker run -itd --name=<name> --net=host -v /data/application/hongzg:/data/application/hongzg -v /mnt/:/mnt -v /hpe_share/:/hpe_share -p 22 -p 8080 -p 8888 --privileged --device=/dev/tcaicard20 --device=/dev/tcaicard21 --device=/dev/tcaicard22 --device=/dev/tcaicard23 --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 300g jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0 /bin/bash
```
其他依赖库
```
-- opencv-python
```
* 打开configs.py对训练参数进行设置：
* 参数说明：

| 参数名              | 说明                                                                              |
|------------------|---------------------------------------------------------------------------------|
| image_path       | 用于训练的数据集的路径                                                                     |
| train_num        | `MSCOCO/train2017`数据集包含**118,287**张图像，设置该参数来确定用于训练的图像的数量                        |
| resume_path      | 默认为None，设置为已经训练好的**权重文件路径**时可对该权重进行继续训练，注意选择的权重要与**gray**参数相匹配                  |
| device           | 模型训练设备 cpu or gpu                                                               |
| batch_size       | 批量大小                                                                            |
| num_workers      | 加载数据集时使用的CPU工作进程数量，为0表示仅使用主进程，（在Win10下建议设为0，否则可能报错。Win11下可以根据你的CPU线程数量进行设置来加速数据集加载） |
| learning_rate    | 训练初始学习率                                                                            |
| num_epochs       | 训练轮数                                                                               |

* 设置完成参数后，运行**run_train.py**即可开始训练：

```python
    # 数据集相关参数
    parser.add_argument('--image_path', default=r'E:/project/Image_Fusion/DATA/RoadScene_dataset', type=str, help='数据集路径')
    parser.add_argument('--gray', default=True, type=bool, help='是否使用灰度模式')
    parser.add_argument('--train_num', default=10000, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--resume_path', default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size, default=4')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=1e-4, help='select the learning rate, default=1e-2')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
```

* 你可以在运行窗口看到类似的如下信息：

```
==================模型超参数==================
----------数据集相关参数----------
image_path: ***************
train_num: 200
----------训练相关参数----------
device: None
device: cpu
batch_size: 4
num_workers: 0
num_epochs: 50
learning rate : 0.0001
==================模型超参数==================
训练数据载入完成...
设备就绪...
Tensorboard 构建完成，进入路径：./runs\train_04-02_14-43\logsepoch=50
然后使用该指令查看训练过程：tensorboard --logdir=./
测试数据载入完成...
initialize network with normal type
网络模型及优化器构建完成...
Epoch [1/50]: 100%|██████████| 50/50 [00:17<00:00,  2.82it/s, learning_rate=0.0001, pixel_loss=0.0314, ssim_loss=0.182]
Epoch [2/50]: 100%|██████████| 50/50 [00:19<00:00,  2.62it/s, learning_rate=9e-5, pixel_loss=0.0762, ssim_loss=0.258]
Epoch [3/50]: 100%|██████████| 50/50 [00:18<00:00,  2.67it/s, learning_rate=8.1e-5, pixel_loss=0.029, ssim_loss=0.249]
Epoch [4/50]: 100%|██████████| 50/50 [00:18<00:00,  2.66it/s, learning_rate=7.29e-5, pixel_loss=0.0424, ssim_loss=0.159]
Epoch [5/50]: 100%|██████████| 50/50 [00:18<00:00,  2.65it/s, learning_rate=6.56e-5, pixel_loss=0.051, ssim_loss=0.232]
Epoch [6/50]: 100%|██████████| 50/50 [00:18<00:00,  2.65it/s, learning_rate=5.9e-5, pixel_loss=0.0403, ssim_loss=0.298]
Epoch [7/50]: 100%|██████████| 50/50 [00:18<00:00,  2.67it/s, learning_rate=5.31e-5, pixel_loss=0.022, ssim_loss=0.26]
Epoch [8/50]: 100%|██████████| 50/50 [00:18<00:00,  2.64it/s, learning_rate=4.78e-5, pixel_loss=0.0449, ssim_loss=0.259]
......
Epoch [45/50]: 100%|██████████| 50/50 [00:19<00:00,  2.62it/s, learning_rate=9.7e-7, pixel_loss=0.0605, ssim_loss=0.256]
Epoch [46/50]: 100%|██████████| 50/50 [00:19<00:00,  2.63it/s, learning_rate=8.73e-7, pixel_loss=0.0549, ssim_loss=0.244]
Epoch [47/50]: 100%|██████████| 50/50 [00:18<00:00,  2.63it/s, learning_rate=7.86e-7, pixel_loss=0.028, ssim_loss=0.291]
Epoch [48/50]: 100%|██████████| 50/50 [00:19<00:00,  2.63it/s, learning_rate=7.07e-7, pixel_loss=0.0456, ssim_loss=0.194]
Epoch [49/50]: 100%|██████████| 50/50 [00:18<00:00,  2.64it/s, learning_rate=6.36e-7, pixel_loss=0.0372, ssim_loss=0.31]
Epoch [50/50]: 100%|██████████| 50/50 [00:18<00:00,  2.65it/s, learning_rate=5.73e-7, pixel_loss=0.0489, ssim_loss=0.216]
Finished Training
训练耗时： 965.3936777114868
Best val loss: 21.221090
```

* Tensorboard查看训练细节：
  * **logs**文件夹下保存Tensorboard文件
  * 进入对于文件夹后使用该指令查看训练过程：`tensorboard --logdir=./`
  * 在浏览器打开生成的链接即可查看训练细节

#### 使用我提供的权重继续训练

* 打开args_fusion.py对训练参数进行设置
* 首先确定训练模式（Gray or RGB）
* 修改**resume_path**的默认值为已经训练过的权重文件路径

* 运行**run_train.py**即可运行



### Fuse Image

* 打开**run_fusion.py**文件，调整**defaults**参数
  * 确定原图像路径和权重路径
  * 确定保存路径
* 运行**run_fusion.py**
* 你可以在运行窗口看到如下信息：

```shell
runs/train_02-22_07-29/checkpoints/epoch025-loss0.000.pth model loaded.
载入数据...
开始融合...
```