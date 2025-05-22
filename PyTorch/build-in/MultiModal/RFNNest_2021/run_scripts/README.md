## 参数介绍

参数名 | 说明 | 默认值
-----------------|-----------------|-----------------
model_name |模型名称。 | --model_name DenseFuse
RFN | 训练阶段，Ture为第二阶段。 | --RFN False
image_path_autoencoder |第一阶段训练数据路径。 | /mnt_qne00/dataset/coco/train2017/
image_path_rfn |第二阶段训练数据路径。 | ../dataset/KAIST
gray |是否使用灰度模式。 | --gray True
train_num| 训练图片数量。 | --train_num 1600
resume_nestfuse| 第一阶段导入已训练好的模型。 | None
resume_rfn| 第二阶段导入已训练好的模型。 | None
batch_size| 批量大小。 | --batch_size 16
num_epochs | 训练轮数。 | --num_epochs 1
lr | 学习率。| --lr 1e-4
num_workers | 载入数据集所调用的cpu线程数 | --num_workers 0
device|训练设备。| --device sdaa