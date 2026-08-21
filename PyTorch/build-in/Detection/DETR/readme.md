```markdown
## 1. 模型链接
- **原始仓库链接**：[MMDetection (GitHub)](https://github.com/open-mmlab/mmdetection/blob/main/README_zh-CN.md)

## 2. 快速开始

使用本模型执行训练的主要流程如下：
1. **基础环境安装**：介绍训练前需要完成的基础环境检查和安装。
2. **获取数据集**：介绍如何准备训练所需的 **COCODogCat** 数据集。
3. **构建环境**：介绍如何构建模型运行所需要的 Python 虚拟环境。
4. **启动训练**：介绍如何运行训练脚本。

### 2.1 基础环境安装

请参考主仓库的基础环境安装章节，完成训练前的基础环境检查和安装（如驱动、固件等）。

### 2.2 准备数据集

#### 2.2.1 获取数据集

本模型使用 **COCODogCat** 数据集。该数据集是一个开源数据集，是从完整的 MS COCO 数据集中筛选出的仅包含 **"Dog" (狗)** 和 **"Cat" (猫)** 两个类别的子集，用于轻量化验证和快速训练。

#### 2.2.2 处理数据集

请确保数据集已下载并解压。根据训练脚本的默认配置，建议将数据集存放在模型目录的上级 `data` 目录中（即 `../data`），或者根据实际路径修改训练命令中的 `--datapath` 参数。

### 2.3 构建环境

所使用的环境下需包含 PyTorch 框架虚拟环境。

1. 执行以下命令，启动虚拟环境（根据实际环境名称修改）：

   ```bash
   conda activate torch_env_py310

```

2. 安装 Python 依赖。确保已安装项目所需的依赖包：
```bash
pip install -r requirements.txt

```


### 2.4 启动训练

1. 在构建好的环境中，进入模型训练脚本所在目录。

2. 运行训练。该模型支持单机单卡训练。
```bash
cd ./DETR

```


执行以下命令启动训练：
```bash
python weloTrainStep.py \
    --name train \
    --model detr \
    --steps 100 \
    --datapath ../data \
    --no-validate \
    --auto-scale-lr \
    --cfg-options optim_wrapper.accumulative_counts=4 \
    --batch-size 8 2>&1 | tee sdaa.log

```
