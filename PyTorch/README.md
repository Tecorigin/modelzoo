# PyTorch

## 介绍

PyTorch是目前主流的开源的AI学习框架，主要用于构建和训练神经网络。提供了简单易用的API，使得定义模型、加载数据和训练模型变得简单。太初元碁对其进行了深度的适配和优化，新增SDAA加速卡和自研软件栈支持，并针对分类、检测、分割、自然语言处理、语音等场景常用的各类经典和前沿的AI模型适配支持。


## 模型适配全流程

您可以将其他硬件平台上训练的模型迁移到本仓库进行训练。适配流程图如下：

![小模型训练全流程适配](http://docs.tecorigin.com/assets/model_train-e54a23d4.jpg)

详细信息，请参考[模型适配指南](./doc/模型适配指南.md)。

## 模型支持列表和链接

| MODELS | Type | Train Mode |Distributed Train|
| ------------- | ------------- | ------------- | ------------- |
| [BERT](./NLP/BERT) | 自然语言处理 | AMP | YES


## Release Note

### v1.0.0:
* 新增BERT模型

## 免责声明
ModelZoo仅提供公共数据集的下载链接。这些公共数据集不属于ModelZoo, ModelZoo也不对其质量或维护负责。请确保您具有这些数据集的使用许可。确保符合其对应的使用许可。

如果您不希望您的数据集公布在ModelZoo上或希望更新ModelZoo中属于您的数据集，请在Github中提交issue,我们将根据您的issue删除或更新您的数据集。衷心感谢您对我们社区的理解和贡献。


## 许可认证
Teco ModelZoo的license是Apache 2.0.具体内容，请参见[LICENSE](../LICENSE)文件。
