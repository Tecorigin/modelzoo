# T5

## 1.模型概述

T5（Text-To-Text Transfer Transformer）是由Google提出的一个基于 Transformer 的文本到文本（Text-to-Text）模型，能够统一处理多种自然语言任务（如翻译、摘要、问答等）通过转换为文本生成问题。来自于[Transformers](https://github.com/huggingface/transformers)，使用wmt14-en-de-pre-processed数据进行训练，训练环境为`transformers`。

## 2.快速开始

### 2.1环境配置

请参考[基础环境安装](https://gitee.com/tecorigin/modelzoo/blob/main/doc/Environment.md)章节，完成训练前的基础环境检查和安装。

```bash
pip install -r requirements.txt
pip install -e .
```

### 2.2数据集获取与准备

数据集来自于Hugging Face的[hub](https://huggingface.co/datasets)，不需要手动下载。但是由于网络问题，可能需要手动下载wmt14-en-de-pre-processed.py。

### 2.3模型训练

运行以下命令进行训练。
```bash
bash examples/pytorch/translation/train_t5.sh
```

### 2.4结果储存

训练过程中的loss如下图所示。

![loss figure](loss.png)

可以看出随着训练的进行loss呈现下降趋势。