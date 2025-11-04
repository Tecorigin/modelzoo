# 参数介绍

| 参数名      | 解释   | 样例                  |
|----------|------|---------------------|
| dataroot | 数据集路径 | --dataroot xxx      |
| name     | 保存模型的名称 | --name HER2         |
| model    |所运行模型名称| --model cycle_gan   |
| netG     |生成器| --netG unet_128     |
|     n_epochs     |稳定学习率的训练轮次（默认50）| --n_epochs          |
|     n_epochs_decay     |学习率逐渐下降的训练轮次（默认50| --n_epochs_decay 50 |

