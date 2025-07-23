### 参数说明

参数名 | 解释    | 样例
-----------------|-------|-----------------
device | 设备数量    | -device 1
total_epochs | 是练总epoch数 | --total_epochs 100
batch_size | 训练时的batch size | --batch_size 4
autocast | 开启混合精度训练 | --autocast True
data_path | 数据集路径 | --data_path /data/teco-data/COCO
early_stop | 在没有明显改善的情况下停止训练 | --early_stop 100
num_workers | 数据加载的线程数量 | --num_workers 8
lr0 | 初始学习率 | --lr0 0.01
warmup_bias_lr | warmup初始学习率 | --warmup_bias_lr 0.1
optimizer | 优化器 | --optimizer auto
imgsz | 输入图片shape | --imgsz 640
mosaic | mosaic增强概率 | --mosaic 1.0