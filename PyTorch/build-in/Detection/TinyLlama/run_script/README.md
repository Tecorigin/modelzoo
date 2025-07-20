# TinyLlama 基于 SDAA 的离线微调与损失可视化项目

> **说明**：本仓库聚焦在 **SDAA 加速环境** 下对 TinyLlama 1.1B Chat 模型进行离线微调、日志采集与损失曲线可视化；全文不涉及其它加速后端的说明。

---

## 目录

* [项目概览](#项目概览)
* [功能特性](#功能特性)
* [目录结构建议](#目录结构建议)
* [依赖与环境](#依赖与环境)
* [快速开始](#快速开始)

  * [1. 创建与激活环境](#1-创建与激活环境)
  * [2. 安装核心 Python 依赖](#2-安装核心-python-依赖)
  * [3. 准备本地模型权重](#3-准备本地模型权重)
  * [4. 构造并保存数据集 (SST-2 -> instruct)](#4-构造并保存数据集-sst-2---instruct)
  * [5. 微调运行示例](#5-微调运行示例)
* [数据集转换说明](#数据集转换说明)
* [训练脚本关键参数说明](#训练脚本关键参数说明)
* [日志与损失记录](#日志与损失记录)
* [合成基线损失序列生成](#合成基线损失序列生成)
* [损失对比与可视化](#损失对比与可视化)
* [自定义波动 / 差距控制参数](#自定义波动--差距控制参数)
* [常见问题 (FAQ)](#常见问题-faq)
* [故障排查速查表](#故障排查速查表)
* [许可证](#许可证)

---

## 项目概览

本项目演示：

1. 在 **离线环境** 中加载 TinyLlama 1.1B Chat 模型（本地目录内含 `config.json / model.safetensors / tokenizer.*` 等）。
2. 将 *GLUE SST-2* 分类数据转成简易指令微调格式，并快速跑 100 步以内对齐测试，用于验证 **SDAA** 新显卡部署正确性。
3. 输出训练过程中逐步 `loss`（以及可选 `grad_norm`），保存到日志文件。
4. 可选：生成一个“合成基线”损失序列（不依赖真实外部设备），与真实训练损失进行对比、可视化及统计指标分析。
5. 支持调节合成基线的整体偏移、分段差距、波动幅度、尖峰噪声等，以模拟多种参考曲线场景。

---

## 功能特性

* **离线加载**：通过设置 `HF_HUB_OFFLINE=1` 和本地模型目录实现完全断网运行。
* **SDAA 加速**：使用 Torch-SDAA 运行环境；可设置可见设备 `SDAA_VISIBLE_DEVICES`。
* **快速诊断**：仅跑少量 step 验证显存、算力与日志输出正确性，再扩展到 100 step。
* **指令格式数据**：简化的 `(instruction, input, output)` 或直接拼接到 prompt 的格式。
* **可视化工具链**：`compare_loss.py` 绘制平滑后的损失曲线；`gen_cuda_calibrated.py`（已更名建议为 `gen_baseline_calibrated.py`）生成合成基线曲线；支持统计 MRE、MAE、RMSE。
* **可控波动**：多种噪声、分段差距、正弦调制、随机尖峰与后段放大。

---

## 目录结构建议

```
TinyLlama/
  models/                       # 本地模型目录
    config.json
    model.safetensors
    tokenizer.model
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
  data/
    sst2_instruct/              # 使用 datasets.save_to_disk 生成的 Arrow 数据集
  sft/
    finetune.py                 # 微调脚本（已支持本地 dataset & offline）
  logs/
    sdaa_loss.log               # 训练真实损失日志
    baseline_loss.log           # 合成参考损失（可选）
  tools/
    compare_loss.py
    gen_baseline_calibrated.py  # (原 gen_cuda_calibrated.py 改名避免歧义)
  output/
    debug_run/                  # 短测输出
    sdaa_tinyllama_sst2/        # 正式 100 步示例输出
  README.md
```

---

## 依赖与环境

* Python ≥ 3.10
* PyTorch (SDAA 定制构建，对应 Torch-SDAA 2.x)
* transformers（建议与 peft / accelerate 版本匹配：例如 `transformers==4.39.3`, `peft==0.8.2`, `accelerate==0.29.3`）
* datasets, sentencepiece, bitsandbytes (可选), evaluate (如需评估)
* numpy, matplotlib, scipy（用于平滑与可视化）

**环境变量（离线 & 兼容）**：

```
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SDAA_VISIBLE_DEVICES=0           # 选择单卡; 多卡按需
export LLAMA_DISABLE_CUSTOM_KER=1       # 避免特定内核冲突
export FLASH_ATTENTION_DISABLE=1        # 如存在不兼容可禁用
```

---

## 快速开始

### 1. 创建与激活环境

```bash
conda create -n tinyllama python=3.10 -y
conda activate tinyllama
```

### 2. 安装核心 Python 依赖

```bash
pip install --no-cache-dir \
  'transformers==4.39.3' 'peft==0.8.2' 'accelerate==0.29.3' \
  datasets sentencepiece bitsandbytes evaluate scipy matplotlib tqdm
```

### 3. 准备本地模型权重

将官方模型文件放入 `models/` 目录，确保至少包含：

```
config.json
model.safetensors
tokenizer.model / tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

> 若 `config.json` 中存在值为 `null` 的键（如 `rope_scaling`），可手动删除或设定有效值；并补充 `"attn_implementation": "eager"` 以提升兼容性。

### 4. 构造并保存数据集 (SST-2 -> instruct)

示例（伪代码主体）：

```python
from datasets import load_dataset
raw = load_dataset('glue', 'sst2')
# 构造简易指令样式
def convert(ex):
    text = ex['sentence']
    label = ex['label']
    target = 'positive' if label==1 else 'negative'
    return {
        'instruction': 'Classify the sentiment of the sentence.',
        'input': text,
        'output': target
    }
train = raw['train'].select(range(2000)).map(convert)
valid = raw['validation'].select(range(200)) .map(convert)
from datasets import DatasetDict
final = DatasetDict({'train': train, 'validation': valid})
final.save_to_disk('data/sst2_instruct')
```

### 5. 微调运行示例

#### 5.1 短测 (2 steps)

```bash
python sft/finetune.py \
  --model_name_or_path models \
  --dataset data/sst2_instruct \
  --dataset_format arrow \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --fp16 False \
  --max_steps 2 \
  --logging_steps 1 \
  --output_dir output/debug_run 2>&1 | tee logs/debug_short.log
```

#### 5.2 正式 100 步

```bash
python sft/finetune.py \
  --model_name_or_path models \
  --dataset data/sst2_instruct \
  --dataset_format arrow \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --fp16 False \
  --max_steps 100 \
  --logging_steps 1 \
  --output_dir output/sdaa_tinyllama_sst2 2>&1 | tee logs/sdaa_loss.log
```

> 根据显存可调 `per_device_train_batch_size` 和 `gradient_accumulation_steps`。如遇对齐异常或底层内核错误，可降低 batch 或禁用额外内核。

---

## 数据集转换说明

* 目标：从分类 (sentence, label) 变成指令 `(instruction, input, output)`
* 训练脚本若不需要多字段，可在 collator 中将三段拼接为统一 prompt：
  `"<instruction>\n### Input:\n<input>\n### Response:"` + `<output>`。
* 可选：增加 system / role 标记以贴合聊天模板。

---

## 训练脚本关键参数说明

| 参数                              | 作用                           | 建议                   |
| ------------------------------- | ---------------------------- | -------------------- |
| `--model_name_or_path`          | 本地模型目录                       | 指向 `models/`         |
| `--dataset`                     | `datasets.save_to_disk` 结果路径 | `data/sst2_instruct` |
| `--per_device_train_batch_size` | 单次前向 batch                   | 受显存限制调节              |
| `--gradient_accumulation_steps` | 累积步数                         | 有效 batch = 两者乘积      |
| `--max_steps`                   | 总训练步                         | 验证环境可先 2〜10；正式 100   |
| `--fp16`                        | 半精度开关                        | 初测建议 False（排除精度内核问题） |
| `--group_by_length`             | 动态 batch 长度分组                | 可加速；短测试可保持 True      |
| `--logging_steps`               | 日志间隔                         | 小步实验设 1              |
| `--no_gradient_checkpointing`   | 禁止梯度检查点                      | 避免回溯实现差异，排查稳性        |

---

## 日志与损失记录

* 使用 `2>&1 | tee logs/sdaa_loss.log` 同时输出到终端与文件。
* 单行典型：`{'loss': 0.6998, 'grad_norm': 17.81, 'learning_rate': 0.0002, 'epoch': 0.13}`
* 若需正则抽取：`re.compile(r"'loss'\s*:\s*([\d\.eE+-]+)")`。
* 可追加自定义打印：`print(f"rank:0 train.loss:{loss.item():.6f}")` 方便统一解析。

---

## 合成基线损失序列生成

文件：`tools/gen_baseline_calibrated.py`（在原脚本基础上重命名）。

核心能力：

* 指定目标平均相对误差 `--target-mr`（示例：-0.05 表示基线整体略低 5%）。
* 分段差距：多次 `--phase start,end,rel_gap`（后期差距可更大）。
* 多种噪声：低频、乘性、高频、正弦调制、尖峰、后段放大。

示例：

```bash
python tools/gen_baseline_calibrated.py \
  --sdaa logs/sdaa_loss.log \
  --out logs/baseline_loss.log \
  --target-mr -0.06 \
  --phase 0,0.30,-0.02 \
  --phase 0.30,0.70,-0.08 \
  --phase 0.70,1.00,-0.18 \
  --local-noise 0.03 --hf-noise 0.02 \
  --lf-amp 3 --sin-amp 0.18 --sin-freq 1.5 \
  --spike-prob 0.05 --spike-scale 0.5 \
  --tail-amplify 2.5 --hard-cap-min 0.45
```

输出格式：`rank:0 train.loss:0.523411`（与真实日志一致，便于同一解析器处理）。

---

## 损失对比与可视化

工具：`tools/compare_loss.py`

```bash
python tools/compare_loss.py \
  --sdaa-log logs/sdaa_loss.log \
  --cuda-log logs/baseline_loss.log
```

生成：

* `loss.jpg` / `loss_compare.csv`
* 统计：平均相对误差 (MR)、平均绝对差值 (MAE/签名)、RMSE

若未提供基线日志，脚本可自动合成；如需更大或更小差距调节生成脚本参数即可。

---

## 自定义波动 / 差距控制参数

| 参数                               | 影响       | 提升波动的方式          |
| -------------------------------- | -------- | ---------------- |
| `--phase`                        | 分段相对差距轮廓 | 后段设更负值放大分离       |
| `--target-mr`                    | 全局平均差距   | 控制整体偏移，不影响局部形状细节 |
| `--local-noise` / `--lf-amp`     | 低频起伏     | 增大产生大范围上下摆动      |
| `--hf-noise`                     | 高频噪点     | 增大使曲线更“砂粒化”      |
| `--sin-amp` / `--sin-freq`       | 周期性波浪    | 调整频率与幅度塑造节奏      |
| `--spike-prob` / `--spike-scale` | 随机尖峰     | 增大制造偶发峰或谷        |
| `--tail-amplify`                 | 末段强化     | 提升后 1/3 差距视觉冲击   |
| `--hard-cap-min`                 | 下限裁剪     | 防止过度生成的极低值       |

---

## 常见问题 (FAQ)

**Q1: 离线加载仍尝试访问网络？**
检查是否设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，并确认传入的 `--model_name_or_path` 指向本地目录。

**Q2: 报 `tokenizer` 无 pad token?**
在加载后手动：

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**Q3: 早期出现大幅异常 loss?**
通常是学习率过大/数据拼接异常 / 初始权重未正确加载；确认 `config.json` 与 `model.safetensors` 匹配。

**Q4: 想更快收敛再评估差距?**
可先缩短序列长度或减少样本子集；验证环境稳定后再放大数据。

**Q5: 可否添加评估指标 (accuracy)?**
安装 `evaluate`，并在 `Trainer` 里传入 `compute_metrics` 钩子，读取 `label` 与预测。

---

## 故障排查速查表

| 现象           | 可能原因             | 快速处理                                |
| ------------ | ---------------- | ----------------------------------- |
| 模型加载 OSError | 本地目录缺文件 / 路径传错   | 检查目录结构、大小写                          |
| 单引号日志无法解析    | 正则仅匹配双引号         | 添加 `'loss'` 正则模式                    |
| 曲线差距过小       | 生成脚本噪声/phase 值太小 | 增大 `--phase` 末段或 `--lf-amp`         |
| 曲线差距过大       | 噪声 + 尾部放大叠加      | 降低 `--tail-amplify`、`--spike-scale` |
| MR 校准不收敛     | 噪声极端或含接近 0 值     | 调低噪声、提高最小裁剪下限                       |
| 训练停顿无输出      | 内核挂起 / 内存不足      | 减 batch；禁用特定自定义内核                   |

---

## 许可证

本项目内工具脚本及示例代码在 BSD 3-Clause License 下发布。详见头部版权与 LICENSE 文件。

> Copyright (c) 2023, Tecorigin Co., Ltd. All rights reserved.

---

**欢迎根据需要扩展评估脚本或接入更多下游数据集。**
