## 参数介绍
参数名 | 解释 | 样例
-----------------|-----------------|-----------------
epoch| 训练轮次 | --epoch 400
batch_size | 每次的batch_size | --batch_size 9
device | 设备类型。 | --device sdaa
d_model | 特征维度 | --d_model 64
n_head | 自注意力头的个数 |  --n_head 4
num_layers | transformer块的个数 | --num_layers 2
dropout | dropout rate | --dropout 0.1
max_length | 位置编码的最大长度 | --max_length 1500
ddp | 是否开启分布式 | --ddp
inference | 是否运行推理 | --inference
nproc_per_node|DDP时，每个node上的rank数量。不输入时，默认为1，表示单SPA运行。|--nproc_per_node 4
nnode|DDP时，node数量。不输入时，默认为1，表示单机运行。|--nnode 1