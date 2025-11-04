#  Waveglow
## 1. 模型概述
WaveGlow是一种基于流的生成网络，专为语音合成设计，能够直接从梅尔频谱图（mel-spectrograms）生成高质量的音频波形。由NVIDIA提出，WaveGlow结合了WaveNet的音质优势和基于流的模型的并行化能力，解决了WaveNet在生成音频时速度慢的问题。它使用了一种称为“仿射耦合”（affine coupling）的变换，通过一系列可逆的操作将简单的概率分布转换为复杂的音频信号分布。WaveGlow模型架构包括多个流模块，每个模块包含卷积层和仿射耦合层，允许高效的训练和推理过程。由于其完全卷积的特性，WaveGlow可以利用GPU进行加速，实现快速的音频生成，使其成为实时应用的理想选择。此外，WaveGlow简化了生成高质量语音的过程，无需复杂的声码器或后处理步骤，极大地提升了语音合成技术的实用性与效率。
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
```angular2html
|-- LJSpeech
    |-- metadata.csv 
    |-- wavs               
        |-- LJ015-0267.wav      
        |-- LJ010-0307.wav      
        |-- LJ010-0300.wav  
        |--........        
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
    cd <ModelZoo_path>/PyTorch/contrib/speech/waveglow
    ```
   
2. 运行训练。
    ```
    python train.py -c config.json
    ```

### 2.5 训练结果
模型训练4h，得到结果如下  
|模型|Epoch|Batch size|Loss|  
| :-: | :-: | :-: | :-: |   
｜waveglow|1|12|-3.90|




