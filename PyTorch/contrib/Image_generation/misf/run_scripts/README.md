## 参数介绍

| 参数名                         | 说明                              | 示例                                          |
| --------------------------- | ------------------------------- | ------------------------------------------- |
| `--seed`                    | 设置随机种子，确保实验结果可复现                | `--seed 10`                                 |
| `--debug`                   | 是否开启调试模式，开启后会输出调试信息             | `--debug 1`                                 |
| `--verbose`                 | 是否开启详细输出模式                      | `--verbose 0`                               |
| `--train_flist`             | 训练集图像路径文本文件                     | `--train_flist ./data/face.txt`             |
| `--val_flist`               | 验证集图像路径文本文件                     | `--val_flist ./data/face.txt`               |
| `--test_flist`              | 测试集图像路径文本文件                     | `--test_flist ./data/face.txt`              |
| `--train_mask_flist`        | 训练集掩码路径文本文件                     | `--train_mask_flist ./data/mask.txt`        |
| `--val_mask_flist`          | 验证集掩码路径文本文件                     | `--val_mask_flist ./data/mask.txt`          |
| `--test_mask_flist`         | 测试集掩码路径文本文件                     | `--test_mask_flist ./data/mask.txt`         |
| `--lr`                      | 学习率                             | `--lr 0.0001`                               |
| `--d2g_lr`                  | 判别器与生成器学习率比例                    | `--d2g_lr 0.1`                              |
| `--beta1`                   | Adam 优化器的 beta1 参数              | `--beta1 0.0`                               |
| `--beta2`                   | Adam 优化器的 beta2 参数              | `--beta2 0.9`                               |
| `--input_size`              | 输入图像尺寸（正方形），0 表示使用原始尺寸          | `--input_size 256`                          |
| `--max_iters`               | 最大训练迭代次数                        | `--max_iters 100`                           |
| `--l1_loss_weight`          | L1 重建损失权重                       | `--l1_loss_weight 1`                        |
| `--fm_loss_weight`          | 特征匹配损失权重                        | `--fm_loss_weight 10`                       |
| `--style_loss_weight`       | 风格损失权重                          | `--style_loss_weight 250`                   |
| `--content_loss_weight`     | 内容（感知）损失权重                      | `--content_loss_weight 0.1`                 |
| `--inpaint_adv_loss_weight` | 对抗损失权重                          | `--inpaint_adv_loss_weight 0.1`             |
| `--gan_loss`                | 对抗损失类型（`nsgan`、`lsgan`、`hinge`） | `--gan_loss nsgan`                          |
| `--gan_pool_size`           | 对抗训练中使用的假样本池大小，0 表示不使用          | `--gan_pool_size 0`                         |
| `--sample_interval`         | 每隔多少次迭代保存一次样本（0 表示不保存）          | `--sample_interval 10`                      |
| `--sample_size`             | 每次采样保存的图像数量                     | `--sample_size 5`                           |
| `--log_interval`            | 日志打印间隔（单位：迭代）                   | `--log_interval 100000`                     |
| `--mask_reverse`            | 是否反转掩码（1 表示反转）                  | `--mask_reverse 0`                          |
| `--mask_threshold`          | 掩码阈值，用于特定数据处理                   | `--mask_threshold 0`                        |
| `--gpu`                     | 使用的 GPU ID 列表                   | `--gpu 0`                                   |
| `--batch_size`              | 训练时每个 batch 的样本数                | `--batch_size 1`                            |
| `--save_interval`           | 每隔多少次迭代保存一次模型                   | `--save_interval 10000`                     |
| `--eval_interval`           | 每隔多少次迭代进行一次模型评估                 | `--eval_interval 2000`                      |
| `--train_sample_interval`   | 训练时的图像保存间隔                      | `--train_sample_interval 1000`              |
| `--eval_sample_interval`    | 评估时的图像保存间隔                      | `--eval_sample_interval 200`                |
| `--train_sample_save`       | 保存训练样本的目录                       | `--train_sample_save ./result/train_sample` |
| `--eval_sample_save`        | 保存评估样本的目录                       | `--eval_sample_save ./result/eval_sample`   |
| `--test_sample_save`        | 保存测试样本的目录                       | `--test_sample_save ./result/test_20`       |
| `--model_load`              | 加载的模型配置名称                       | `--model_load celebA_InpaintingModel`       |

