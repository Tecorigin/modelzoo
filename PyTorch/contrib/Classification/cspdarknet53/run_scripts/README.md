## 参数介绍

参数名 | 说明 | 示例
-----------------|-----------------|-----------------
data | 数据集路径。 | /mnt/nvme/common/train_dataset/mini-imagenet
epoch| 训练轮次，和训练步数冲突。 | --epoch 120
step | 训练步数，和训练轮数冲突。 | --step 1000
batch_size/bs | 每个rank的batch_size。 | --batch_size 32 / --bs 32
rank | DDP时，每个node上的rank数量。不输入时，默认为1，表示单SPA运行。 | --rank 4
world-size| number of nodes for distributed training。| --world-size 1
dist-url| url used to set up distributed training | --dist-url 'tcp://127.0.0.1:65501'
dist-backend| distributed backend | --dist-backend 'tccl'
distributed | 是否开启DDP | --distributed True
multiprocessing-distributed | Use multi-processing distributed training to launch N processes per node which has N GPUs. | --multiprocessing-distributed
lr |学习率。| --lr 1e-3
wd |权重衰减。| --wd 1e-4
gpu | 使用的gpu号 | --gpu 0
workers | dataloader读取数据进程数。 | --workers 2
