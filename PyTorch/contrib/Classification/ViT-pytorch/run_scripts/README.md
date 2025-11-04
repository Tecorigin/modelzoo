## 参数介绍

参数名 | 说明 | 示例
-----------------|-----------------|-----------------
name |自定义名称。 | --name cifar10-100_500
dataset |数据集名称。 | --dataset cifar10
model_type |模型类型。|--model_type ViT-B_16
pretrained_dir | 预训练权重路径。 | --pretrained_dir checkpoint/ViT-B_16.npz
train_batch_size | 批次大小。 | --train_batch_size 64
fp16_opt_level | 混合精度类型。 | --fp16_opt_level 01
num_steps | 训练轮次。 | --num_steps 100