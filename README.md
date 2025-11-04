简体中文

# ModelZoo
## 简介
该仓库提供了一套基于SDAA加速卡的易于模型训练、推理的SDAA社区，以满足AI开发者和用户的多样化需求。其注重性能、功能和准确性，使用户能够轻松地扩展和定制模型，以适用于各种应用场景，通过在SDAA加速卡上运行适配的模型实现最佳的可重复精度和性能。此外，该仓库还提供了针对特定行业的端到端解决方案，确保AI技术的无缝集成和部署。

## 仓库介绍

ModelZoo仓库包含2个子仓库，包含Pytorch、PaddlePaddle、TecoInference，您可以直接克隆本仓库，或者分别克隆需要的子仓库，并根据子仓库的README指引进行使用。


| 子仓库  | 说明 |
| ------------- | ------------- |
| [PyTorch](./PyTorch) | 基于PyTorch-SDAA框架的训练模型集合 |
| [PaddlePaddle](./PaddlePaddle) | 基于PaddlePaddle-SDAA框架的训练模型集合 |
| [TecoInference](./TecoInference) | 基于TecoInferenceEngine框架的推理模型集合 |


## 模型训练支持列表和链接

| 模型 | 类型 | 训练模式 |是否使用分布式训练|
| ------------- | ------------- | ------------- | ------------- |
| [ResNet](./PaddlePaddle/Classification/ResNet) | 图像分类 |AMP|是
| [BERT](./PyTorch/NLP/BERT) | 自然语言处理 | AMP | 是

## 模型推理支持列表和链接

| 模型 | 推理卡数 |精度模式|
| ------------- | ------------- | ------------- |
| [ResNet](./TecoInference/example/classification/resnet/README.md) |单卡|FP16

## 发版记录

V1.1.0：
* 新增ResNet推理模型

V 1.0.0：
* 新增ResNet和BERT训练模型。



## 免责声明
ModelZoo仅提供公共数据集的下载链接。这些公共数据集不属于ModelZoo, ModelZoo也不对其质量或维护负责。请确保您具有这些数据集的使用许可。基于这些数据集的模型仅可用于非商业研究和教育。

致数据集所有者：

如果您不希望您的数据集公布在ModelZoo上或希望更新ModelZoo中属于您的数据集，请在Github/Gitee中提交issue,我们将根据您的issue删除或更新您的数据集。衷心感谢您对我们社区的理解和贡献。

## 许可认证
Teco ModelZoo采用The 3-Clause BSD License。具体内容，请参见[LICENSE](./LICENSE)文件。
