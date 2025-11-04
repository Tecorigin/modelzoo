# OC_SORT

## 1.模型概述
OC_SORT强调“观察”在恢复已丢失轨迹和降低丢失期间线性运动模型积累的误差时的作用，它保持简单，在线和实时，但提高了对遮挡和非线性运动的鲁棒性。在MOT17和MOT20上分别达到了63.2和62.1的HOTA，达到了SOTA。它还在KITTI行人跟踪数据集和DanceTrack数据集等目标运动高度非线性的数据集上实现新的技术水平。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

OC_SORT运行在MOT17数据集上，数据集配置可以参考:
```
wget https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip
unzip MOT17.zip


```

### 2.3 启动训练

准备环境：
```
pip install -r requirements.txt
```
确保安装后环境中只有一个版本的numpy，如果不是可以在执行命令前先执行“pip uninstall -y numpy”进行卸载。

模型单机单卡训练：
```
export PADDLE_XCCL_BACKEND=sdaa
export PADDLE_DISTRI_BACKEND=tccl
export SDAA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --log_dir=scripts --gpus 0,1,2,3 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp 2>&1|tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练总耗时为2小时，得到结果如下：

| 加速卡数量 | 模型 |Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | OC_SORT | 64 | Epochs:36 | 1.288211 | / |