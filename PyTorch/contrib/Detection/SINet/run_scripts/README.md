## 参数介绍
| 参数名               | 说明                                       | 示例                                                                |
| ----------------- | ---------------------------------------- | ----------------------------------------------------------------- |
| `--epoch`         | 总训练 epoch 数                              | `--epoch 40`                                                      |
| `--lr`            | 初始学习率                                    | `--lr 1e-4`                                                       |
| `--batchsize`     | 每个 batch 的样本数（注意：每张图像大约占用 500MB 显存）      | `--batchsize 36`                                                  |
| `--trainsize`     | 训练图像的尺寸（图像将被缩放到 `trainsize × trainsize`） | `--trainsize 352`                                                 |
| `--clip`          | 梯度裁剪上限（用于防止梯度爆炸）                         | `--clip 0.5`                                                      |
| `--decay_rate`    | 每次学习率衰减的比例                               | `--decay_rate 0.1`                                                |
| `--decay_epoch`   | 每隔多少个 epoch 执行一次学习率衰减                    | `--decay_epoch 30`                                                |
| `--gpu`           | 指定使用的 GPU 编号（仅适用于单 GPU 训练）               | `--gpu 1`                                                         |
| `--save_epoch`    | 每隔多少个 epoch 保存一次模型权重                     | `--save_epoch 10`                                                 |
| `--save_model`    | 模型保存路径                                   | `--save_model ./Snapshot/2020-CVPR-SINet/`                        |
| `--train_img_dir` | 训练图像的文件夹路径                               | `--train_img_dir /data/teco-data/COD10K_CAMO/TrainDataset/Image/` |
| `--train_gt_dir`  | 训练标签（Ground Truth）的文件夹路径                 | `--train_gt_dir /data/teco-data/COD10K_CAMO/TrainDataset/GT/`     |
