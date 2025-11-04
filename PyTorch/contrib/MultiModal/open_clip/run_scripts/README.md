# Activity

add-open-clip All activity All users All time Showing most recent first


---

## 🚀 常用运行参数说明（部分）

| 参数名                    | 类型    | 默认值         | 描述                                          |
| ---------------------- | ----- | ----------- | ------------------------------------------- |
| `--train-data`         | str   | None        | 训练集路径，支持 CSV / WebDataset，多个源可用 `::` 分隔     |
| `--val-data`           | str   | None        | 验证集路径                                       |
| `--dataset-type`       | str   | `auto`      | 数据类型（`webdataset`、`csv`、`synthetic`、`auto`） |
| `--batch-size`         | int   | 64          | 每 GPU 的训练批次大小                               |
| `--epochs`             | int   | 32          | 总训练轮数                                       |
| `--lr`                 | float | 自动推导        | 学习率，若未显式指定将根据模型自动推导                         |
| `--model`              | str   | `RN50`      | 模型名称，如 `ViT-B-32`、`RN50x4` 等                |
| `--pretrained`         | str   | `''`        | 预训练模型路径或 HuggingFace / ModelZoo 中的 tag      |
| `--precision`          | str   | `amp`       | 训练精度（`amp`、`fp16`、`bf16`、`fp32`）            |
| `--logs`               | str   | `./logs/`   | 日志输出路径                                      |
| `--resume`             | str   | None        | 指定 checkpoint 路径以恢复训练                       |
| `--cache-dir`          | str   | None        | 模型或分词器缓存目录                                  |
| `--save-frequency`     | int   | 1           | 每多少轮保存一次模型                                  |
| `--device`             | str   | `sdaa`      | 使用设备类型（`cuda`、`sdaa`、`cpu`）                 |
| `--dist-url`           | str   | None        | 启动分布式训练的 URL 地址                             |
| `--wandb-project-name` | str   | `open-clip` | 使用 wandb 时的项目名称                             |
| `--siglip`             | flag  | False       | 是否启用 SigLip 风格的损失函数                         |
| `--distill-model`      | str   | None        | 蒸馏时指定教师模型结构                                 |
| `--remote-sync`        | str   | None        | 设置远程 checkpoint 自动同步目录路径                    |

---

## 🧪 示例：使用 CSV 数据训练 ViT 模型

```bash
python run_open_clip.py \
  --train-data /workspace/data/train.csv \
  --dataset-type csv \
  --model ViT-B-32 \
  --epochs 10 \
  --batch-size 128 \
  --precision amp