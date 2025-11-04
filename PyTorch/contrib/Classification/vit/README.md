 # Classification-ViT：Vision Transformer在Pytorch当中的实现

## 1、模型概述
Vision Transformer 是将Transformer应用在图像分类的模型
## 2、快速开始
使用本模型执行训练的主要流程如下：

1.运行环境配置：介绍训练前需要完成的运行环境配置和检查。

2.数据集准备：介绍如何使用如何获取并处理数据集。

3.启动训练：介绍如何运行训练。
### 2.1、运行环境配置
#### 2.1.1 拉取代码仓
```python
git clone https://gitee.com/tecorigin/modelzoo.git
```

#### 2.1.2 创建Teco虚拟环境
```python
cd /modelzoo/PyTorch/contrib/faster_net
conda activate torch_env

pip install -r requirements.txt

# install tcsp_dllogger
git clone https://gitee.com/xiwei777/tcap_dllogger.git
cd tcap_dllogger
python setup.py install
```
### 2.2、数据集准备
#### 2.2.1 数据集介绍
本次分类任务所用的是猫狗数据集，训练集包含50000张猫和狗的图片
#### 2.2.2 数据集下载
数据集下载地址如下，里面已经包括了训练集、测试集，无需再次划分：  
链接: https://pan.baidu.com/s/1hYBNG0TnGIeWw1-SwkzqpA
提取码: ass8
#### 2.2.3 数据集处理
```python
python txt_annotation.py
# 在准备好数据集后，需要在根目录运行txt_annotation.py生成训练所需的cls_train.txt
```

### 2.3、启动训练
训练命令：支持单机单SPA以及单机单卡（DDP）。训练过程保存的权重以及日志均会保存在"logs"中。\
- 单机单SPA训练
    ```
    python run_scripts/run_train.py --model_name vit_b_16 --batch_size 32 --lr 1e-2 --device sdaa --epoch 200 --distributed False --use_amp True --train_annotation_path datasets/cls_train.txt --val_pairs_path datasets/cls_test.txt
    ```
- 单机单卡训练（DDP）
    ```
    python run_scripts/run_train.py --model_name vit_b_16 --nproc_per_node 3 --batch_size 32 --lr 1e-2 --device sdaa --epoch 200 --distributed True --use_amp True --train_annotation_path datasets/cls_train.txt --val_pairs_path datasets/cls_test.txt
    ```
    训练命令参数说明参考[文档](run_scripts/README.md)。

### 2.4 训练结果
训练条件

| 芯片   | 卡 | 模型          | 混合精度 | batch size |
|------|---|-------------|------|------------|
| SDAA | 1 | vit_b_16 | 是    | 8          |

训练结果量化指标如下表所示

| 加速卡数量 | epoch | 混合精度 | batch size | accuracy |
|-------|-------|------|------------|---------|
| 1     | 200    | 是    | 8          | 0.686   |
