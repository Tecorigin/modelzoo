#  CLIP
## 1. 模型概述
CLIP（Contrastive Language-Image Pre-training）是由OpenAI开发的深度学习模型，结合了自然语言处理和计算机视觉的能力。通过大规模的图文对数据进行训练，CLIP将图像和文本投影到相同的嵌入空间，实现跨模态对比学习。CLIP在图像分类、图像检索、文本生成图像和零样本学习等任务中表现出色，尤其是在零样本学习方面无需专门训练即可通过自然语言描述完成新任务。其强大的跨模态学习能力使CLIP在计算机视觉和自然语言处理领域具有广泛的应用前景和深远影响，成为深度学习领域的重要模型之一。

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


#### 数据集和权重目录结构：
```angular2html
|-- data
    |-- bpe_simple_vocab_16e6.txt.gz  #词汇表文件
    |-- checkpoints         # 权重放置地方
    |-- datasets               # 你需要的数据集，可选择下载
        |-- flickr8k-images       # 数据集图片文件
        |-- en_train.json         # 训练集标注
        |-- en_val.json           # 测试集标注
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
    cd <ModelZoo_path>/PyTorch/contrib/MultiModal/clip
    ```
   
2. 运行训练。该模型支持单机单卡。

- 单机单卡四SPA
   ```
   python run_scripts/run_clip.py --model_name clip --nproc_per_node 4 --batch_size 32 --lr 1e-6 --epoch 20 --device sdaa --checkpoint_path path/clip/data/checkpoints/ViT-B-16-OpenAI.pth --datasets_path path/clip/data/datasets/
   ```
  
- 单机单卡单SPA
   ```angular2html 
   python run_scripts/run_clip.py --model_name clip --batch_size 32 --lr 1e-6 --epoch 50 --autocast True --device sdaa --checkpoint_path path/clip/data/checkpoints/ViT-B-16-OpenAI.pth --datasets_path path/clip/data/datasets/
   ```

  更多训练参数参考[README](run_scripts/README.md)



### 2.5 训练结果

**训练loss曲线:**

![epoch_loss.png](img%2Fepoch_loss.png)

**召回曲线：**

![epoch_recall.png](img%2Fepoch_recall.png)


**指标结果：**

| 加速卡数量 | 模型   | 混合精度 | Batch_size | shape     | epoch | Text R Mean(%) | Image R Mean(%) | Overall Mean(%) |
|-------|------|------|------------|-----------|-------|----------------|-----------------|-----------------|
| 1     | clip | AMP  | 32         | 224 x 224 | 50    | 94.07          | 87.75           | 90.91           |

[//]: # (| Metric     | Text &#40;%&#41; | Image &#40;%&#41; |)

[//]: # (|------------|-----------|-----------|)

[//]: # (| R@1        | 85.7      | 71.78     |)

[//]: # (| R@5        | 97.3      | 94.12     |)

[//]: # (| R@10       | 99.2      | 97.34     |)

[//]: # (| R Mean     | 94.07     | 87.75     |)

[//]: # (| Overall Mean | 90.91  | 90.91     |)









### Reference
https://gitcode.com/bubbliiiing/clip-pytorch/overview

https://github.com/openai/CLIP   

https://github.com/alibaba/AliceMind  