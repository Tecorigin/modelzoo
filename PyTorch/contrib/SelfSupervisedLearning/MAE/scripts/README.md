## 示例命令

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 ../main_pretrain.py \
    --batch_size 64 \
    --epochs 1 \
    --accum_iter 1 \
    --model mae_vit_large_patch16 \
    --input_size 224 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --weight_decay 0.05 \
    --blr 1.5e-4 \
    --min_lr 0.0 \
    --warmup_epochs 40 \
    --data_path /data/datasets/imagenet \
    --output_dir /data/ckpt/mae_pretrain \
    --device sdaa \
    --seed 0 \
    --resume '' \
    --start_epoch 0 \
    --num_workers 10 \
    --pin_mem \
    --world_size 1 \
    --local_rank -1 \
    --dist_url env:// 2>&1 | tee train_sdaa_3rd.log
```


## 参数说明

下表列出了脚本中可用的主要参数，包含参数名称、描述、默认值以及示例用法。可根据自身需求进行修改。

| 参数                | 描述                                                                 | 默认值                      | 示例                                      |
|---------------------|----------------------------------------------------------------------|----------------------------|-------------------------------------------|
| **--batch_size**    | 每个 GPU/进程上的 batch size。                                      | 64                         | `--batch_size 128`                        |
| **--epochs**        | 训练的总轮数（epoch 数）。                                          | 400                        | `--epochs 200`                           |
| **--accum_iter**    | 梯度累加步数。相当于在不增加显存占用的前提下扩大有效 batch size。    | 1                          | `--accum_iter 2`                          |
| **--model**         | 训练的模型名称，需与脚本支持的 MAE 模型对应（如 `mae_vit_large_patch16`）。 | mae_vit_large_patch16      | `--model mae_vit_base_patch16`            |
| **--input_size**    | 模型输入图像的尺寸（如 224 表示 224×224）。                        | 224                        | `--input_size 224`                        |
| **--mask_ratio**    | MAE 预训练中被遮挡的 patch 的比例。                                 | 0.75                       | `--mask_ratio 0.75`                       |
| **--norm_pix_loss** | 若指定该参数，则使用归一化后的像素进行损失计算（每个 patch 归一化）。| False（未指定则为 False）  | `--norm_pix_loss` (启用)                  |
| **--weight_decay**  | 优化器的 weight decay（L2 正则化）系数。                            | 0.05                       | `--weight_decay 0.05`                     |
| **--lr**            | 绝对学习率。如果不指定，则脚本内部会根据 `blr` 和实际 batch size 来计算学习率。 | None                       | `--lr 1e-4`                               |
| **--blr**           | 基准学习率（Base LR）。当 `--lr` 未指定时，会根据批大小等因素自动缩放。 | 1e-3                       | `--blr 1.5e-4`                            |
| **--min_lr**        | 学习率调度（如 cosine decay）中的最小学习率。                        | 0.0                        | `--min_lr 1e-5`                           |
| **--warmup_epochs** | 学习率预热（Warmup）的轮数。                                         | 40                         | `--warmup_epochs 10`                      |
| **--data_path**     | 训练数据集的根目录（如 ImageNet）；需包含 `train`、`val` 等子目录。    | /data/datasets/imagenet    | `--data_path /path/to/imagenet`           |
| **--output_dir**    | 用于保存检查点（checkpoint）和日志的目录。                           | /data/ckpt/mae_pretrain    | `--output_dir /path/to/output`            |
| **--device**        | 训练所用的设备（`cuda` / `sdaa` / `cpu` 等）。在 TecoPyTorch 上需设为 `sdaa`。 | sdaa                       | `--device sdaa`                           |
| **--seed**          | 随机种子，便于结果复现。                                            | 0                          | `--seed 42`                               |
| **--resume**        | 若需从某个检查点继续训练，指定该检查点路径；默认为空字符串表示不恢复。 | ''                         | `--resume /path/to/checkpoint.pth`        |
| **--start_epoch**   | 若从检查点恢复训练，则从哪个 epoch 开始计数。                         | 0                          | `--start_epoch 50`                        |
| **--num_workers**   | 数据加载（DataLoader）的 worker 线程数。                             | 10                         | `--num_workers 16`                        |
| **--pin_mem**       | 是否在数据加载中使用 pinned memory（可加速 GPU 传输）。              | True（默认启用）           | `--pin_mem`（启用），`--no_pin_mem`（禁用） |
| **--world_size**    | 分布式训练时的总进程数（所有节点上的进程之和）。                      | 1                          | `--world_size 8`                          |
| **--local_rank**    | 当前进程在本地节点上的 rank（由 torchrun 内部自动设置）。             | -1                         | `--local_rank 0`                          |
| **--dist_on_itp**   | 内部分布式启动方式的标志位；使用 `torchrun` 时通常无需手动设置。       | store_true（不常用）       | `--dist_on_itp`                           |
| **--dist_url**      | 用于分布式训练初始化的 URL，一般在使用 `torchrun` 时为 `env://`。      | env://                     | `--dist_url tcp://127.0.0.1:3456`         |

---

## 分布式训练说明

- 使用 `torchrun --standalone --nnodes=1 --nproc_per_node=4`：
  - `--standalone` 表示无需联系其它节点的主服务器进行协调。  
  - `--nnodes=1` 表示只有一个节点（一台机器）。  
  - `--nproc_per_node=4` 表示在该节点上启动 4 个进程（通常对应 4 张 GPU）。  
- 若需要多节点或更多 GPU，请调整相应的参数（如 `--nnodes` 或 `--nproc_per_node` 以及 `--world_size`）。

---

## 日志与检查点

- 检查点会自动保存到 `--output_dir` 指定的文件夹下，默认每 20 个 epoch 保存一次，并在训练结束时保存最终模型。  
- 训练日志（epoch 级别的指标）会追加到该文件夹下的 `log.txt`。  
- 示例命令中的 `2>&1 | tee train_sdaa_3rd.log` 用于将所有终端输出（包括错误信息）保存到 `train_sdaa_3rd.log` 文件中，便于之后查看。

