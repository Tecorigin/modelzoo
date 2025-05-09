## 参数介绍

参数名 | 说明 | 示例
-----------------|-----------------|-----------------
model_name |模型名称。 | --model_name resnet50
epoch| 训练轮次，和训练步数冲突。 | --epoch 50
step| 训练步数 | --step 100
batch_size/bs | 每个rank的batch_size。 | --batch_size 32 / --bs 32
nproc_per_node | DDP时，每个node上的rank数量。不输入时，默认为1，表示单SPA运行。 | --nproc_per_node 4
nnode | DDP时，node数量。不输入时，默认为1，表示单机运行。| --nnode 2
node_rank|多机时，node的序号。|--node_rank 0
master_addr|多机时，主节点的IP地址。|--master_addr 127.0.0.1
master_port|多机时，主节点的端口号。|--master_port 129500
device|设备类型。|--device sdaa