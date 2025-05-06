# Tecorigin ModelZoo仓库模型适配及合入指南

## 1. 概述

Tecorigin ModelZoo是基于太初加速卡训练或推理的模型合集。基于仓库中的模型，用户能够轻松地扩展和定制模型，通过在太初加速卡上运行适配的模型实现最佳的精度和性能。如果适配后的模型满足精度和性能要求，您可以通过PR（Pull Request）方式将模型提交到Tecorigin ModelZoo仓库。

本指南介绍如何适配模型以及如何将适配的模型提交到Tecorigin ModelZoo仓库。
    

## 2. 整体流程

基于Tecorigin ModelZoo仓库，适配模型及合入模型的整体流程如下：

![image](./contri_flow.png#pic_center)

1. **检查Tecorign ModelZoo开发环境检查**：了解Tecorign ModelZoo的开发环境，熟悉模型开发使用的框架信息以及加速卡硬件信息等，确保开发环境能够充分满足当前任务的需求。详情内容，请参考[检查Tecorign ModelZoo开发环境](./01-检查开发环境.md)。
2. **适配模型**：将GPU设备上运行的模型训练代码迁移到太初加速卡上运行。详情内容，请参考[适配模型](./02-适配模型.md)。
3. **准备模型提交所需的文件**：添加运行接口目录、README文件（模型使用说明）、Requirement文件（第三方依赖安装文件）等文件，确保模型满足Tecorigin ModelZoo仓库及开源使用要求。详情内容，请参考[准备模型合入文件](./03-准备模型合入文件.md)。
4. **提交PR**：对合入文件进行检查，并通过PR（Pull Request）将相应内容提交到Tecorigin ModelZoo仓库。详情内容，请参考[提交PR](./04-提交PR.md)。

