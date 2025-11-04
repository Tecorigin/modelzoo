参数名 | 说明 | 示例
-----------------|-----------------|-----------------
data_path | 数据集路径（支持 ImageNet、CIFAR-10、CIFAR-100） | --data_path /data/teco-data/imagenet
batch_size | 训练批次大小 | --batch_size 64
epochs | 训练轮次（默认 300） | --epochs 1
lr | 初始学习率 | --lr 0.1
save_path | 模型保存路径 | --save_path ./checkpoints
num_steps | 训练步数（默认 3100） | --num_steps 100
growthRate | DenseNet 的增长率 | --growthRate 12
dropRate | Dropout 比率 | --dropRate 0.2
reduction | Transition 层的通道压缩比率 | --reduction 0.5
bottleneck | 是否使用瓶颈结构 | --bottleneck
depth | DenseNet 的深度（例如 121, 169, 201） | --depth 121
dataset | 使用的数据集（cifar10, cifar100, imagenet） | --dataset imagenet
optMemory | 内存优化级别 | --optMemory 2
cudnn | cuDNN 模式（deterministic 或 default） | --cudnn deterministic