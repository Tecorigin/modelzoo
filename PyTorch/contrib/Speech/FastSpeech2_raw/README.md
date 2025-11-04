#  FastSpeech2
## 1. 模型概述
FastSpeech2 是一种改进的端到端语音合成模型，它基于 Transformer 架构构建，采用了无注意力机制的编码器 - 解码器结构。通过引入长度调节器、对抗训练以及改进的声码器，有效解决了传统语音合成模型中速度慢、韵律不自然等问题。FastSpeech2 模型在语音合成任务中有着卓越的表现，能够快速且稳定地生成高质量、自然流畅的语音。其在训练效率和推理速度上的优势，使其在实时语音合成等场景中极具应用价值。凭借出色的性能和高效的架构，FastSpeech2 成为语音合成领域中具有重要影响力的先进模型之一。
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
LJSpeech数据集下载地址如下
链接: https://keithito.com/LJ-Speech-Dataset/ 

#### 2.2.2 处理数据集
解压数据集，将其按格式放入文件datasets中，格式目录如工作目录所示

#### LJSpeech数据集和权重目录结构：
文件目录结构：
```angular2html
  |--lispeech_raw
          |-- LJSpeech-1.1
          |-- output
```
step 1.修改config中preprocess.yaml文件的corpus_path、raw_path

step 2.运行
  ```
  python3 prepare_align.py config/LJSpeech/preprocess.yaml
  ```

step 3.下载ljspeech的对应文件MFA，放在preprocessed_data/LJSpeech/TextGrid/
  ```
  https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4
  ```

step 4.运行：
  ```
  python3 preprocess.py config/LJSpeech/preprocess.yaml
  ```

step 5.解压pth文件
  ```
  cd hifigan
  unzip generator_LJSpeech.pth.tar.zip
  unzip generator_universal.pth.tar.zip 
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
    cd <ModelZoo_path>/PyTorch/contrib/Speech/FastSpeech2
    ```
   
2. 运行训练。

   ```
   torchrun --nproc_per_node 4 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml

   ```

### 2.5 训练结果
模型训练4h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: |   
|4|FastSpeech2|是|121|16|5.21|

