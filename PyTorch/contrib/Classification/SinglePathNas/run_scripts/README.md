参数名 | 说明 | 示例
-------|------|------
exp_name | 实验名称 | --exp_name spos_c10_train_supernet
layers | 超网络层数 | --layers 20
num_choices | 每层候选操作数 | --num_choices 4
batch_size | 训练批次大小 | --batch_size 64
epochs | 训练轮次 | --epochs 600
num_steps | 每轮训练步数 | --num_steps 100
lr | 初始学习率 | --lr 0.025
momentum | 动量 | --momentum 0.9
weight-decay | 权重衰减 | --weight-decay 3e-4
print_freq | 训练日志打印频率 | --print_freq 100
val_interval | 验证和保存模型的频率 | --val_interval 5
save_path | 模型检查点保存路径 | --save_path ./checkpoints/
seed | 训练随机种子 | --seed 0
data_path | 数据集目录 | --data_path data/teco-data/
classes | 数据集类别数 | --classes 10
dataset | 数据集名称 | --dataset cifar10
cutout | 是否使用cutout数据增强 | --cutout
cutout_length | cutout增强的裁剪长度 | --cutout_length 16
auto_aug | 是否使用自动数据增强 | --auto_aug
resize | 是否调整图像大小 | --resize