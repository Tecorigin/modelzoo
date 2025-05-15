## 参数介绍

参数名 | 说明 | 示例
-----------------|-----------------|-----------------
model_name |模型名称。 | --model_name Dehamer
dataset-name |使用的数据集。 | --dataset-name NH
train-dir| 训练集路径。 | --train-dir ./../data/train
valid-dir| 验证集路径。 | --valid-dir ./../data/valid
ckpt-save-path | checkpoint保存路径。 | --ckpt-save-path ./ckpts
nb-epochs | 训练轮数。 | --nb-epochs 2
batch-size | 批量大小。| --batch-size 2
train-size|训练集图片尺寸。|--train-size 192 288
valid-size|验证集图片尺寸。|--valid-size 192 288
loss |损失函数。|--loss l1
cuda|是否启用cuda(sdaa)。|--cuda