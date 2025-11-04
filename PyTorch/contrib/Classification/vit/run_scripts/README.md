# 参数介绍

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
model_name | 模型名称 | --model_name vit_b_16
distributed | 是否开启DDP| --distributed True
nproc_per_node | 每个节点上的进程数| --nproc_per_node 3
nnodes | 多机节点数| --nnodes 1
node_rank | 多机节点序号| --node_rank 0
device | 运行设备 | --device sdaa
master_addr | 多机主节点ip地址| --master_addr 192.168.1.1
master_port | 多机主节点端口号| --master_port 29505
use_amp | 是否使用amp | --use_amp True
epochs| 训练总epoch数 |--epochs 200
lr| 学习率 |--lr 1e-3
batch_size| 训练时的batch size |--batch_size 4
train_data_path | 训练数据路径 |--train_data_path datasets
train_annotation_path | 训练标注路径 |--train_annotation_path cls_train.txt
val_data_path| 测试数据路径 |--val_data_path lfw
val_pairs_path| 测试数据标注路径 |--val_pairs_path model_data/lfw_pair.txt

