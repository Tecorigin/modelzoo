## 参数介绍
| 参数名              | 说明                             | 示例                               |
| ---------------- | ------------------------------ | -------------------------------- |
| `--experiment`   | 实验名称，通常用于指定输出文件夹名或实验配置组        | `--experiment peta`               |
| `--approach`     | 使用的模型方法名或方法配置名（如模型架构、算法）       | `--approach inception_iccv`      |
| `--epochs`       | 训练的总 epoch 数                   | `--epochs 60`                    |
| `--batch_size`   | 每个 batch 的训练样本数量               | `--batch_size 32`                |
| `--lr`           | 初始学习率（可用 `--learning-rate` 替代） | `--lr 0.0001`                    |
| `--optimizer`    | 优化器类型，如 `adam`、`sgd` 等         | `--optimizer adam`               |
| `--momentum`     | SGD 优化器使用的动量参数                 | `--momentum 0.9`                 |
| `--weight_decay` | 权重衰减（L2 正则化）系数                 | `--weight_decay 0.0005`          |
| `--start-epoch`  | 从哪个 epoch 开始训练（用于断点续训）         | `--start-epoch 0`                |
| `--print_freq`   | 每训练多少个 batch 打印一次训练状态          | `--print_freq 100`               |
| `--save_freq`    | 每隔多少个 epoch 保存一次模型权重           | `--save_freq 10`                 |
| `--resume`       | 从某个模型文件路径加载 checkpoint 继续训练    | `--resume checkpoints/model.pth` |
| `--decay_epoch`  | 学习率衰减发生的 epoch 点（可设置多个）        | `--decay_epoch "(20, 40)"`       |
| `--prefix`       | 文件名前缀，用于保存文件时添加额外标识            | `--prefix run1_`                 |
| `--evaluate`     | 是否仅运行验证（开启后不会进行训练），作为测试模型用     | `--evaluate`                     |

