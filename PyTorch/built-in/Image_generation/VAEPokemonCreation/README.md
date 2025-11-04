## 模型简介：
通过在700余张宝可梦图像中学习宝可梦图片的分布，实现变分自编码器（VAE），并可视化学习后得到的效果。

## 数据集准备：
由于本仓库使用数据集为课程中提供的40x40的宝可梦图像数据集，所以无需额外准备数据集。
unrar x image.rar 解压缩数据集即可。

## 模型训练：
训练启动：
按照如下指令可以训练过程，并通过Tensorboard实时查看生成效果。
```bash
python main.py
```
特别的，可以通过命令行参数设置如下种种训练参数，尤其是`latent_size`默会影响后续可视化的效果。默认值请参考[main.py](main.py)中的`get_args_parser`函数：
```
options:
  -h, --help            show this help message and exit
  --hidden_size HIDDEN_SIZE
                        VAE settings, size of hidden layer for encoder and decoder
  --latent_size LATENT_SIZE
                        VAE settings, size of the latent vector.
  --batch_size BATCH_SIZE
                        Batchsize per GPU
  --output_dir OUTPUT_DIR
                        output dir for ckpt and logs
  --epoch EPOCH         Number of epochs
  --lr LR               Learning rate
  --device DEVICE       Device: cuda or GPU
  --test_period TEST_PERIOD
                        Test when go through this epochs
  --save_period SAVE_PERIOD
                        masked rate of the input images
  --warmup_epochs WARMUP_EPOCHS
                        warmup epochs
  --min_lr MIN_LR       min lr for lr schedule
  --seed SEED           random seed init
```

  - Tensorboard监视训练过程：
    - 在安装了Tensorboard之后，在当前工作路径中使用如下命令，并通过浏览器打开对应本地域名即可实现监视训练过程：
      ```bash
      tensorboard --logdir ./
      ```