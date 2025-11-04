## 参数介绍

参数名 | 说明 | 示例
-----------------|-----------------|-----------------
data | 数据集路径。 | /mnt/nvme/common/train_dataset/mini-imagenet
epoch| 训练轮次，和训练步数冲突。 | --epoch 2
step | 训练步数，和训练轮数冲突。 | --step 10
batch_size/bs | 每个rank的batch_size。 | --batch_size 4 / --bs 4
rank | DDP时，每个node上的rank数量。不输入时，默认为1，表示单SPA运行。 | --rank 4
world-size| number of nodes for distributed training。| --world-size 1
dist-url| url used to set up distributed training | --dist-url 'tcp://127.0.0.1:65501'
dist-backend| distributed backend | --dist-backend 'tccl'
distributed | 是否开启DDP | --distributed True
multi-gpu | Use multi-processing distributed training to launch 'N processes per node, which has N GPUs | --multi-gpu
model | 模型名 | --model dpn92
gpu | 使用的gpu号 | --gpu 0
workers | dataloader读取数据进程数。 | --workers 2