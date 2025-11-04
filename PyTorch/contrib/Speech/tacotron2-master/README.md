#  Tacotron2
## 1. 模型概述
Tacotron2 是一种端到端的语音合成模型，采用了带有注意力机制的编码器 - 解码器结构，通过引入位置敏感注意力机制和可训练的 WaveNet 声码器，以解决语音合成中韵律不准确和语音质量不佳的问题。Tacotron2 模型在语音合成任务中表现出色，能够生成自然流畅、高度还原人声的语音。由于其卓越的性能和良好的可扩展性，Tacotron2 已成为语音合成领域的重要基础模型之一。
当前支持的 Tacotron2 模型在不同的数据集和应用场景下，通过调整参数和优化训练方式，展现出了稳定且优秀的语音合成能力。

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
    cd <ModelZoo_path>/PyTorch/contrib/speech/tacotron2
    ```
   
2. 运行训练。

- 单卡训练
   ```
   python3 train.py --output_directory=outdir --log_directory=logdir
   ```
  
- 多卡+amp
   ```
   python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,use_amp=True
   ```




### 2.5 训练结果
模型训练4h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: |   
|4|tacotron2|是|1|32|4.189767|




