## 参数介绍：

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
work_dir |实验工作目录 | --work_dir experiments/deeplabv1
data_root| 数据集根目录 | --data_root ./datasets/data
dataset | 数据集名称，可根据用户需求扩展 | --dataset vaihingen
num_classes | 数据集目标类别数，vaihingen为6 | --num_classes 6
model_name | 模型名称 | --model_name /deeplabv1
distributed | 是否开启DDP| --distributed True
nproc_per_node | 每个节点上的进程数| --nproc_per_node 4
nnodes | 多机节点数| --nnodes 1
node_rank | 多机节点序号| --node_rank 0
master_addr | 多机主节点ip地址| --master_addr 192.168.1.1
master_port | 多机主节点端口号| --master_port 29505
use_amp | 是否使用amp | --use_amp True
default_rank| 默认进程号，主要用于DDP训练时指定主进程 |--default_rank 0
test_only| 是否开启测试模式 |--test_only True
total_epochs| 训练总epoch数 |--total_epochs 150
optimizer| 优化器 |--optimizer sgd
lr| 学习率 |--lr 0.01
lr_policy| 学习率调整策略 |--lr_policy cosine
crop_val| 是否裁切后评估 |--crop_val False
batch_size| 训练时的batch size |--batch_size 4
val_batch_size| 评估时的batch size | --val_batch_size 1
crop_size| 图像裁切尺寸 | --crop_size 512
continue_training| 是否断点重训 | --val_batch_size 1
ckpt| 断点重训加载的checkpoint文件 | --ckpt experiments/example/best_deeplabv1_vaihingen.pth
loss_type| 选择loss类型 | --loss_type sgd
device| 选择device类型 | --device sdaa
weight_decay| 设置weight_decay | --weight_decay 0.0001
random_seed| 设置random_seed| --random_seed 1
print_interval| 输出日志间隔的iteration数 | --print_interval 5
val_epoch| 在验证集上进行评估间隔的epoch数 | --val_epoch 5