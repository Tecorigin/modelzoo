# BERT Text Summarization

## 1.模型概述

由于BERT不是Seq2Seq模型，无法进行Text Summarization任务，选择Transformers库支持的、同时也和BERT相似的BART模型。其使用编码器-解码器架构，通过去噪自编码预训练，学习恢复被扰乱的文本，使其在生成摘要时能够更好地理解和压缩输入内容。来自于[Transformers](https://github.com/huggingface/transformers)，使用cnn_dailymail数据进行训练，训练环境为`transformers`。

## 2.快速开始

### 2.1环境配置

请参考[基础环境安装](https://gitee.com/tecorigin/modelzoo/blob/main/doc/Environment.md)章节，完成训练前的基础环境检查和安装。

```bash
pip install -r requirements.txt
pip install -e .
```

### 2.2数据集获取与准备

数据集来自于Hugging Face的[hub](https://huggingface.co/datasets)，不需要手动下载。

### 2.3模型训练

运行以下命令进行训练。
```bash
bash examples/pytorch/summarization/train_summarization.sh
```

### 2.4结果展示

训练过程中的loss如下图所示。

![loss figure](loss.png)

可以看出随着训练的进行loss呈现下降趋势。