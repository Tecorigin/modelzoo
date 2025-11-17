## 参数介绍

| 参数名                 | 说明                                                     | 示例                                  |
| ------------------- | ------------------------------------------------------ | ----------------------------------- |
| `--dataroot`        | 数据集路径参数，指向包含子文件夹的数据集根目录。        | `--dataroot /data/teco-data/maps` |
| `--name`            | 实验名称，用于在 `checkpoints/` 下保存模型和日志目录。                    | `--name maps_cyclegan`       |
| `--gpu_ids`         | 使用的 GPU ID，逗号分隔。`-1` 表示使用 CPU。                         | `--gpu_ids 0,1` 或 `--gpu_ids -1`    |
| `--checkpoints_dir` | 模型和结果的保存根路径。                                           | `--checkpoints_dir ./checkpoints`   |
| `--print_freq`      | 每隔多少个训练 step 打印一次日志（控制台输出）。                            | `--print_freq 1`                  |
| `--model`           | 模型类型，可选值有：`cycle_gan`、`pix2pix`、`test`、`colorization`。       | `--model cycle_gan`                 |
| `--n_epochs`       | 初始学习率训练的 epoch 数（未衰减阶段）。                                                        | `--n_epochs 100`       |
| `--max_iters`      | 最大训练迭代步数，设为 `-1` 表示不限制（按 epoch 控制）。                                         | `--max_iters 100`       |