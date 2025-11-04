# TecoInferenceEngine
## 1. 简介
TecoInferenceEngine（小模型）是基于TVM高度适配T1芯片的推理框架，基于AI编译技术，为您提供高效的小模型推理解决方案。本仓库提供了TecoInferenceEngine（小模型）推理框架支持的模型推理示例。针对每个模型，我们提供了详细的推理步骤说明，介绍如何通过TecoInferenceEngine（小模型）推理框架快速运行推理，满足开发者和用户的推理需求。

## 2. 模型列表

| 模型 | 精度模式 | 推理卡数 |
| ------------- | ------------- | ------------- |
|[ResNet](./example/classification/resnet/README.md) | FP16 | 单卡 |

## 3. 基础环境安装

请参考[基础环境安装](./doc/Environment.md)，完成使用前的基础环境安装和检查，之后点击模型列表中对应的模型链接来查看具体的使用教程。

## 4. 发版记录


**V0.1.0**
- 新增ResNet50模型推理,精度对齐TensorRT推理结果

## 5. 免责声明
ModelZoo仅提供公共数据集的下载链接。这些公共数据集不属于ModelZoo, ModelZoo也不对其质量或维护负责。请确保您具有这些数据集的使用许可。确保符合其对应的使用许可。

如果您不希望您的数据集公布在ModelZoo上或希望更新ModelZoo中属于您的数据集，请在Github/Gitee中提交issue,我们将根据您的issue删除或更新您的数据集。衷心感谢您对我们社区的理解和贡献。


## 6. 许可认证
Teco ModelZoo的license是Apache 2.0.具体内容，请参见[LICENSE](../LICENSE)文件。