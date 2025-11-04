#  BLIP
## 1. 模型概述
BLIP（Bootstrapped Language-Image Pretraining） 是一种先进的多模态预训练模型，专注于图像和文本的联合学习。通过引入自引导机制和跨模态注意力机制，BLIP 在处理视觉与语言任务时表现出色，能够生成高质量的文本描述，并准确理解图像内容。BLIP 模型在多个下游任务中展示了卓越的性能，如图像字幕生成、视觉问答（VQA）、文本到图像检索等。

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
coco2017数据集下载地址如下：
captions_train2017.json、captions_val2017.json
链接: https://huggingface.co/datasets/merve/coco/tree/main/annotations

下载train2017数据、下载val2017数据
链接：https://cocodataset.org/#download
#### 2.2.2 处理数据集
解压数据集，将其按格式放入文件datasets中，格式目录如工作目录所示

#### LJSpeech数据集和权重目录结构：
```angular2html
|-- coco2017
    |-- images
      |-- train2017
            |-- 000000000001.png     
            |-- 000000000001.jpg  
            |-- 000000000002.png   
            |-- 000000000002.jpg 
            |--.......
      |-- val2017               
            |-- 000000000011.png     
            |-- 000000000011.jpg  
            |-- 000000000021.png   
            |-- 000000000021.jpg  
            |--.......
    |-- annotations
      |-- captions_train2017.json
      |-- captions_val2017.json
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
    cd <ModelZoo_path>/PyTorch/contrib/MultiModal/blip
    ```
   
2. 运行训练。
单卡训练
   ```
   python3 train_caption.py --is_use_amp=False --distributed=False
   或：
   sh /script/train_sdaa_single.sh

   ```
  
- 多卡+amp
   ```
   python -m torch.distributed.run --nproc_per_node=4 train_caption.py 
   或：
   sh /script/train_sdaa_multi.sh
   ```

### 2.5 训练结果
模型训练6h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: |   
|4|BLIP|是|5|6|3.20|




### Reference
https://github.com/salesforce/BLIP
