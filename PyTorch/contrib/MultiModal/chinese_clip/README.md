#  Chinese-CLIP
## 1. 模型概述
Chinese-CLIP 是一个基于 OpenAI 的 CLIP 模型的中文版本。CLIP（Contrastive Language-Image Pretraining）是一种由 OpenAI 发布的多模态学习模型，能够同时理解图像和文本之间的关系。Chinese-CLIP 扩展了这一模型，使其适用于中文语境，可以处理中文文本和图像之间的对应关系。
## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装
请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
flickr8k数据集下载地址如下，里面已经包括了训练集、验证集（与测试集一样），无需再次划分：

链接: https://pan.baidu.com/s/1UzaGmbEGz1BXZ0IXK1TT7g    提取码: exg3

#### 2.2.2 处理数据集
解压数据集，将其按格式放入文件datasets中，格式目录如工作目录所示

#### 2.2.3 获取权重 (ViT-B-16-OpenAI.pth)
链接: https://pan.baidu.com/s/1b9Nt-UuqOJfhbhJYVyrK0g 提取码: mfnc

#### 2.2.4 获取bert for chinese权重(bert-from-clip-chinese-1M)
https://huggingface.co/YeungNLP/bert-from-clip-chinese-1M 


#### 数据集和权重目录结构：
```angular2html
|-- data
    |-- checkpoints   # 权重放置地方
        |-- bert-from-clip-chinese-1M   # chinese-bert权重
        |-- ViT-B-16-OpenAI.pth # vit权重
    |-- datasets               # 你需要的数据集，可选择下载
        |-- flickr8k-images       # 数据集图片文件
        |-- cn_train.json         # 训练集标注
        |-- cn_val.json           # 测试集标注
```

### 2.3 构建环境
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
   
2. 所需安装的包
    ```
    pip install -r requirements.txt
    ```
    
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/MultiModal/chinese_clip
    ```
   
2. 运行训练。该模型支持单机单卡。

-  单机单卡四SPA
   ```
   python run_scripts/run_chineseclip.py --model_name chinese_clip --nproc_per_node 4 --batch_size 64 --lr 1e-5 --epoch 40 --device sdaa --checkpoint_path path/chinese_clip/data/checkpoints/ViT-B-16-OpenAI.pth --bert_ckpt_path path/chinese_clip/data/checkpoints/bert-from-clip-chinese-1M --datasets_path path/chinese_clip/data/datasets/
   ```
     
- 单机单卡单SPA
   ```angular2html 
   python run_scripts/run_chineseclip.py --model_name chinese_clip --batch_size 32 --lr 1e-6 --epoch 40 --autocast True --device sdaa --checkpoint_path path/chinese_clip/data/checkpoints/ViT-B-16-OpenAI.pth --bert_ckpt_path path/chinese_clip/data/checkpoints/bert-from-clip-chinese-1M --datasets_path path/chinese_clip/data/datasets/
   ```
  更多训练参数参考[README](run_scripts/README.md)



### 2.5 训练结果

**训练loss曲线:**

![epoch_loss.png](img%2Fepoch_loss.png)

**召回曲线：**

![epoch_recall.png](img%2Fepoch_recall.png)


**指标结果：**

| 加速卡数量 | 模型           | 混合精度 | Batch_size | shape     | epoch | Text R Mean(%) | Image R Mean(%) | Overall Mean(%) |
|-------|--------------|------|------------|-----------|-------|----------------|-----------------|-----------------|
| 1     | chinese_clip | AMP  | 32         | 224 x 224 | 40    | 57.46          | 49.89           |   53.68         |

[//]: # (| Metric     | Text &#40;%&#41; | Image &#40;%&#41; |)

[//]: # (|------------|-----------|-----------|)

[//]: # (| R@1        | 31.2     | 24.1     |)

[//]: # (| R@5        | 63.8      | 55.42     |)

[//]: # (| R@10       | 76.9      | 70.04     |)

[//]: # (| R Mean     | 57.30    | 49.85    |)

[//]: # (| Overall Mean | 53.57  | 53.57     |)




### Reference
https://gitcode.com/bubbliiiing/clip-pytorch/overview

https://github.com/openai/CLIP   

https://github.com/alibaba/AliceMind  