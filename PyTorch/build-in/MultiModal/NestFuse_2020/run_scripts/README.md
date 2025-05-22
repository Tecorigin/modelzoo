## 参数介绍

参数名 | 说明 | 默认值
-----------------|-----------------|-----------------
model_name |模型名称。 | --model_name DenseFuse
image_path |训练数据路径。 | /mnt_qne00/dataset/coco/train2017/
gray |是否使用灰度模式。 | --gray True
train_num| 训练图片数量。 | --train_num 1600
batch_size| 批量大小。 | --batch_size 16
num_epochs | 训练轮数。 | --num_epochs 1
batch-size | 批量大小。| --batch-size 2
lr | 学习率。| --lr 1e-4
resume_path | 是否导入已经训练好的模型。|--resume_path None
num_workers | 载入数据集所调用的cpu线程数 | --num_workers 0
device|训练设备。| --device sdaa