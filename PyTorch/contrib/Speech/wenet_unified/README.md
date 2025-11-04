# U2++ Conformer
## 1. 模型概述
U2++ Conformer是一种先进的端到端语音识别模型，它创新性地融合了卷积神经网络（CNN）、循环神经网络（RNN）和自注意力机制，形成独特的Conformer模块，有效解决了传统语音识别模型在处理长时依赖和复杂声学特征时的难题。U2++ Conformer模型在各类语音识别任务中表现卓越，例如实时语音转文字、语音指令识别以及嘈杂环境下的语音处理等。凭借其强大的性能和对复杂语音场景的高度适应性，U2++ Conformer已成为语音识别领域极具影响力的基础模型之一。
当前支持的 U2++ Conformer 模型涵盖多种配置，能依据不同的应用需求和数据特点，灵活调整网络结构与参数设置，以实现最佳的语音识别效果

## 2. 快速开始
使用本模型执行训练的主要流程如下：  
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。  
2. 获取数据集：介绍如何获取训练所需的数据集。  
3. 启动训练：介绍如何运行训练。  
### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。
### 2.2 准备数据集
#### 2.2.1 获取数据集
AISHELL-1数据集下载地址如下
链接: https://www.openslr.org/33/

#### 2.2.2 处理数据集
解压数据集，将其按格式放入文件datasets中，格式目录如工作目录所示

#### LJSpeech数据集和权重目录结构：
```angular2html
|-- AISHELL-1
    |-- resource_aishell               
        |-- lexicon.txt
        |-- speaker.info    

    |-- data_aishell 
        |-- transcript
            |-- aishell_transcript_v0.8.txt
        |-- wav
            |-- dev
            |-- train
            |-- test
            |-- S001.tar.gz
            |-- ......
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
    cd <ModelZoo_path>/PyTorch/contrib/speech/u2++_conformer/examples/aishell/s0
    ```
   
2. 运行训练。

- 数据处理
   ```
   sh run_u2++_conformer.sh --stage 0 --stop_stage 0 # AIShell-1数据组织成两个文件：wav.scp和text（位于s0/data/目录下）
    # dev为训练过程中的交叉验证，local为中间结果可忽略，test为测试，train为训练
   ```
  
- 标准化文本标签
   ```
   sh run_u2++_conformer.sh --stage 1 --stop_stage 1
   ```
- 创建字典
   ```
   sh run_u2++_conformer.sh --stage 2 --stop_stage 2
   ```
- 将数据转换为模型训练所需的格式
   ```
   sh run_u2++_conformer.sh --stage 3 --stop_stage 3
   ```
- 训练
   ```
   sh run_u2++_conformer.sh --stage 4 --stop_stage 4
   ```
### 2.5 训练结果
模型训练6h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|Acc|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |   
|4|U2++_conformer|是|10|16|25.67%|70.186|
