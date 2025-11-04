### 参数说明

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
model_name | 模型名称 | --model_name efficientnet
distributed | 是否开启DDP| --distributed True
nproc_per_node | 每个节点上的进程数| --nproc_per_node 4
device | 运行设备 | --device sdaa
autocast | 是否使用amp | --autocast True
epochs| 训练总epoch数 |--epochs 100
step| 训练的最大step数 |--step 10
lr| 学习率 |--lr 0.2
batch_size| 训练时的batch size |--batch_size 64