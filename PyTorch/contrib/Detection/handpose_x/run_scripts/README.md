## 参数介绍

| 参数名                 | 说明                      | 示例                                     |
| ------------------- | ----------------------- | -------------------------------------- |
| `--seed`            | 设置随机种子，确保实验结果可复现        | `--seed 126673`                        |
| `--model_exp`       | 模型输出保存的路径               | `--model_exp ./model_exp`              |
| `--model`           | 所使用的模型结构名称              | `--model ReXNetV1`                     |
| `--num_classes`     | 预测点数（关键点数 × 2）          | `--num_classes 42`                     |
| `--GPUS`            | 使用的 GPU 编号列表（字符串格式）     | `--GPUS 0`                             |
| `--train_path`      | 训练集路径（含标注信息）            | `--train_path /data/teco-data/handpose_datasets_v1/` |
| `--pretrained`      | 是否使用 ImageNet 预训练权重     | `--pretrained True`                    |
| `--fintune_model`   | 微调使用的模型路径               | `--fintune_model ./model.pth`          |
| `--loss_define`     | 损失函数类型选择，例如：`wing_loss` | `--loss_define wing_loss`              |
| `--init_lr`         | 初始学习率                   | `--init_lr 0.001`                      |
| `--lr_decay`        | 学习率衰减系数                 | `--lr_decay 0.1`                       |
| `--weight_decay`    | 权重衰减（正则项系数）             | `--weight_decay 1e-6`                  |
| `--momentum`        | 动量参数                    | `--momentum 0.9`                       |
| `--batch_size`      | 每个 batch 的训练样本数量        | `--batch_size 16`                      |
| `--dropout`         | dropout 概率，用于防止过拟合      | `--dropout 0.5`                        |
| `--epochs`          | 最大训练轮数                  | `--epochs 3000`                        |
| `--num_workers`     | 加载数据时的线程数量              | `--num_workers 10`                     |
| `--img_size`        | 输入图像的尺寸（宽 × 高）          | `--img_size (256,256)`                 |
| `--flag_agu`        | 是否开启数据增强                | `--flag_agu True`                      |
| `--fix_res`         | 是否固定图像宽高比               | `--fix_res False`                      |
| `--clear_model_exp` | 是否清空模型输出文件夹             | `--clear_model_exp True`               |
| `--log_flag`        | 是否保存训练日志                | `--log_flag True`                      |

