# QAnything PDF
## 1. 模型概述
QAnything PDF解析模型是一款基于多模态架构的智能文档处理系统，通过融合视觉分析与语义理解技术，实现高精度、结构化的PDF内容提取与知识重构。

### 1.1 BGE-M3 模型说明
BGE-M3 是一个强大的多模态嵌入模型，在本项目中用于：
- 提取文档的语义特征
- 生成文本和图像的统一表示
- 增强文档问答能力

- 仓库链接：https://www.modelscope.cn/models/netease-youdao/QAnything-pdf-parser/files
## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
QAnything PDF解析模型使用 squad 数据集，该数据集为开源数据集，可从 [squad](https://www.modelscope.cn/datasets/modelscope/squad) 下载。
#### 2.2.2 获取 BGE-M3 模型文件
1. 从以下地址下载必要的模型文件：
   - 主要模型文件：[BGE-M3 main branch](https://huggingface.co/BAAI/bge-m3/tree/main)
   - 微调训练文件：[BGE-M3 specific version](https://huggingface.co/BAAI/bge-m3/tree/29fbe26ba08e5b2a5f06bac3632195bc85d3d690)
2. 重要说明：
   - 对于 PyTorch < 2.6 的环境，必须使用 `model.safetensors` 而不是 `pytorch_model.bin`
   - 需要下载的关键文件包括：
     ```
     config.json
     model.safetensors (如果 PyTorch < 2.6)
     pytorch_model.bin (如果 PyTorch >= 2.6)
     special_tokens_map.json
     tokenizer.json
     tokenizer_config.json
     ```
### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    ```
### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/QAnything/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python train.py --train_file /data/teco-data/squad/qanything_train.jsonl --model_name_or_path /data/bigc-data/lsq/QAnything/bge-m3 --output_dir ./outputs/emb_m3_sdaa --batch_size 6 --max_steps 100 --lr 1e-5 2>&1 |tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: 10.337974182595529
MeanAbsoluteError: 0.0038391446984314824
Rule,mean_absolute_error 0.0038391446984314824
fail mean_relative_error=10.337974182595529 <= 0.05 or mean_absolute_error=0.0038391446984314824 <= 0.0002

## 3. 注意事项
1. 模型文件准备：
   - 在运行 test.sh 之前，确保已下载完整的 BGE-M3 模型文件
   - 根据 PyTorch 版本选择正确的模型文件格式（safetensors 或 bin）
   - 检查所有必需的配置文件是否存在

2. 环境要求：
   - PyTorch 版本兼容性检查
   - 确保 CUDA 环境正确配置
   - 建议使用推荐的依赖版本