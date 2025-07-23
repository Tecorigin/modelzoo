### 参数说明

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
model_name | 模型名称 | --model_name /facenet
distributed | 是否开启DDP| --distributed True
nproc_per_node | 每个节点上的进程数| --nproc_per_node 3
device | 运行设备 | --device sdaa
use_amp | 是否使用amp | --use_amp True
epochs| 训练总epoch数 |--total_epochs 400
lr| 学习率 |--lr 1e-1
batch_size| 训练时的batch size |--batch_size 32
weights| 预训练模型 |--weights None
freeze-layers| 除head外，其他权重全部冻结 |--freeze-layers False