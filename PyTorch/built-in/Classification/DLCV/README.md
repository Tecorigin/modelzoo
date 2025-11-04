## 项目概述
本仓库围绕食品图像分类，运用集成模型和多种训练技术。

## 代码运行步骤

### 下载模型：
运行 bash get_checkpoints.sh 下载并解压模型（约 5GB）。

### 训练模型：
运行 bash train.sh <top food data folder> 训练集成模型（可选）。
TTA 测试：运行 bash test_TTA.sh <top food data folder> checkpoints 生成测试时增强输出（可选）。
集成与生成日志：运行 bash test_ensemble.sh <top food data folder> checkpoints 进行集成并生成 Kaggle 日志。