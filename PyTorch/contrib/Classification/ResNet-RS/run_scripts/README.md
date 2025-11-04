## 参数介绍

参数名 | 说明 | 示例
-----------------|-----------------|-----------------
epochs| 训练轮次。 | --epochs 100
batch_size | 训练批次大小 | --batch_size 32
precision | 训练过程中使用的精度 | --precision float32
device | 设备类型 | --device sdaa
max_step | 模型最大迭代步数 | --max_step 100
dataset_path | 数据集存储路径 | --dataset_path /data/teco-data/imagenet
pretrained_path | 预训练权重加载路径 | --pretrained_path /root/checkpoints/a.pth
save_path | 模型权重保存路径 | --save_path /root/checkpoints
lr | 模型学习率 | --lr 0.1
lrf | 模型学习率调整速度 | --lrf 0.1
weight_decay | 权重衰减 | --weight_decay 1e-4
momentum | 动量 | --momentum 0.9
amp | 是否使用混合精度训练 | --amp
model_name | 创建的模型类型 | --model_name resnetrs50 