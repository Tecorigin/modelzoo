## 运行样例

#### 功能测试
1. 单卡/多卡
```
单卡
python run_scripts/run_train_timm.py --data-dir ./imagenet --model efficientnet_b0 -b 64 --sched step --epochs 20 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.3 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016 --nproc_per_node 4
```

#### 长训
1. 单卡/多卡
```
单卡
python run_scripts/run_train_timm.py --data-dir ./imagenet --model efficientnet_b0 -b 64 --sched step --epochs 20 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.3 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016 --nproc_per_node 4

单机四卡
python run_scripts/run_train_timm.py --data-dir ./imagenet --model efficientnet_b0 -b 64 --sched step --epochs 20 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.3 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016 --nproc_per_node 16

单机八卡
python run_scripts/run_train_timm.py --data-dir ./imagenet --model efficientnet_b0 -b 64 --sched step --epochs 20 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-4 --weight-decay 1e-5 --drop 0.3 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016 --nproc_per_node 32
```


### 参数介绍
参数名 | 解释 | 样例 
-----------------|-----------------|----------------- 
nproc_per_node | 每个节点的进程数 | --nproc_per_node 1
nnode | 节点数量 | --nnode 1
data | 数据集路径 (已弃用，请使用 --data-dir) | /path/to/dataset
data-dir | 数据集根目录路径 | --data-dir /path/to/dataset
dataset | 数据集类型和名称 | --dataset imagenet/train
train-split | 数据集训练集分割名称 | --train-split train
val-split | 数据集验证集分割名称 | --val-split validation
train-num-samples | 训练集样本数 | --train-num-samples 50000
val-num-samples | 验证集样本数 | --val-num-samples 5000
dataset-download | 允许下载数据集 | --dataset-download
class-map | 类别到索引映射文件路径 | --class-map /path/to/class_map.txt
input-img-mode | 数据集图像转换模式 | --input-img-mode RGB
input-key | 输入图像键值 | --input-key image
target-key | 目标标签键值 | --target-key label
model | 训练模型名称 | --model resnet50
pretrained | 使用预训练模型 | --pretrained
pretrained-path | 预训练模型路径 | --pretrained-path /path/to/pretrained_model.pth
initial-checkpoint | 加载的检查点路径 | --initial-checkpoint /path/to/checkpoint.pth
resume | 恢复模型和优化器状态的检查点路径 | --resume /path/to/checkpoint.pth
no-resume-opt | 防止从检查点恢复优化器状态 | --no-resume-opt
num-classes | 标签类别数 | --num-classes 1000
gp | 全局池化类型 | --gp avgmax
img-size | 图像大小 | --img-size 224
in-chans | 输入通道数 | --in-chans 3
input-size | 输入图像尺寸 | --input-size 3 224 224
crop-pct | 验证集图像中心裁剪百分比 | --crop-pct 0.875
mean | 数据集像素均值 | --mean 0.485 0.456 0.406
std | 数据集像素标准差 | --std 0.229 0.224 0.225
interpolation | 图像调整插值类型 | --interpolation bilinear
batch-size/-b | 训练批量大小 | -b 128
validation-batch-size/-vb | 验证批量大小 | --validation-batch-size 64
channels-last | 使用 channels_last 内存布局 | --channels-last
fuser | 选择 jit fuser | --fuser te
grad-accum-steps | 累积梯度的步数 | --grad-accum-steps 4
grad-checkpointing | 启用梯度检查点 | --grad-checkpointing
fast-norm | 启用实验性快速归一化 | --fast-norm
model-kwargs | 模型关键字参数 | --model-kwargs dropout 0.2
head-init-scale | 头部初始化比例 | --head-init-scale 0.01
head-init-bias | 头部初始化偏置值 | --head-init-bias 0.1
torchscript | torchscript整个模型 | --torchscript
torchcompile | 启用指定后端的编译 | --torchcompile inductor
device | 使用的设备 | --device cuda
amp | 使用混合精度训练 | --amp
amp-dtype | 低精度 AMP 数据类型 | --amp-dtype float16
amp-impl | 使用的 AMP 实现 | --amp-impl native
no-ddp-bb | 禁用 native DDP 的广播缓冲区 | --no-ddp-bb
synchronize-step | 每步同步 | --synchronize-step
local_rank | 本地排名 | --local_rank 0
device-modules | 设备后端模块 | --device-modules module1 module2
opt | 优化器 | --opt sgd
opt-eps | 优化器 Epsilon | --opt-eps 1e-8
opt-betas | 优化器 Betas | --opt-betas 0.9 0.999
momentum | 优化器动量 | --momentum 0.9
weight-decay | 权重衰减 | --weight-decay 0.0001
clip-grad | 梯度裁剪 | --clip-grad 0.1
clip-mode | 梯度裁剪模式 | --clip-mode value
layer-decay | 层间学习率衰减 | --layer-decay 0.1
opt-kwargs | 优化器关键字参数 | --opt-kwargs lr 0.001
sched | 学习率调度器 | --sched cosine
sched-on-updates | 更新时应用学习率调度器 | --sched-on-updates
lr | 学习率 | --lr 0.1
lr-base | 基本学习率 | --lr-base 0.1
lr-base-size | 基本学习率批量大小 | --lr-base-size 256
lr-base-scale | 学习率与批量大小缩放 | --lr-base-scale linear
lr-noise | 学习率噪声 | --lr-noise 0.5 0.9
lr-noise-pct | 学习率噪声百分比 | --lr-noise-pct 0.67
lr-noise-std | 学习率噪声标准差 | --lr-noise-std 1.0
lr-cycle-mul | 学习率周期长度倍增器 | --lr-cycle-mul 1.0
lr-cycle-decay | 每个学习率周期的衰减量 | --lr-cycle-decay 0.5
lr-cycle-limit | 学习率周期限制 | --lr-cycle-limit 1
lr-k-decay | 余弦/多项式学习率的 k-decay | --lr-k-decay 0.1
warmup-lr | 热身学习率 | --warmup-lr 1e-5
min-lr | 最小学习率 | --min-lr 0
epochs | 训练轮次 | --epochs 300
epoch-repeats | 训练轮次重复倍数 | --epoch-repeats 0.5
start-epoch | 手动指定的起始轮次 | --start-epoch 0
decay-milestones | 多步学习率衰减里程碑 | --decay-milestones 90 180 270
decay-epochs | 学习率衰减周期 | --decay-epochs 90
warmup-epochs | 热身周期 | --warmup-epochs 5
warmup-prefix | 排除热身期从衰减计划中 | --warmup-prefix
cooldown-epochs | 冷却周期 | --cooldown-epochs 0
patience-epochs | Plateau学习率调度器的耐心周期 | --patience-epochs 10
decay-rate/dr | 学习率衰减率 | --decay-rate 0.1
no-aug | 禁用所有训练数据增强 | --no-aug
train-crop-mode | 训练裁剪模式 | --train-crop-mode random
scale | 随机调整尺度 | --scale 0.08 1.0
ratio | 随机调整长宽比 | --ratio 0.75 1.33
hflip | 水平翻转概率 | --hflip 0.5
vflip | 垂直翻转概率 | --vflip 0.0
color-jitter | 色彩抖动因子 | --color-jitter 0.4
color-jitter-prob | 应用任何色彩抖动的概率 | --color-jitter-prob 0.5
grayscale-prob | 应用随机灰度转换的概率 | --grayscale-prob 0.2
gaussian-blur-prob | 应用高斯模糊的概率 | --gaussian-blur-prob 0.1
aa | 使用AutoAugment策略 | --aa v0
aug-repeats | 数据增强重复次数 | --aug-repeats 2
aug-splits | 数据增强分割数量 | --aug-splits 2
jsd-loss | 启用Jensen-Shannon Divergence + 交叉熵损失 | --jsd-loss
bce-loss | 启用二元交叉熵损失 | --bce-loss
bce-sum | 在使用BCE损失时对类别求和 | --bce-sum
bce-target-thresh | 用于二元交叉熵损失的目标阈值 | --bce-target-thresh 0.5
bce-pos-weight | 二元交叉熵损失的正权重 | --bce-pos-weight 0.8
reprob | 随机擦除概率 | --reprob 0.1
remode | 随机擦除模式 | --remode const
recount | 随机擦除计数 | --recount 3
resplit | 不对第一个（干净）增强分割进行随机擦除 | --resplit
mixup | mixup参数 | --mixup 0.2
cutmix | cutmix参数 | --cutmix 0.1
cutmix-minmax | cutmix最小/最大比率 | --cutmix-minmax 0.2 0.8
mixup-prob | 应用mixup或cutmix的概率 | --mixup-prob 0.8
mixup-switch-prob | 当mixup和cutmix都启用时切换到cutmix的概率 | --mixup-switch-prob 0.5
mixup-mode | 应用mixup/cutmix参数的方式 | --mixup-mode batch
mixup-off-epoch | 在此轮次后关闭mixup | --mixup-off-epoch 100
smoothing | 标签平滑参数 | --smoothing 0.1
train-interpolation | 训练插值 | --train-interpolation bilinear
drop | 丢弃率 | --drop 0.5
drop-connect | 丢弃连接率 | --drop-connect 0.2
drop-path | 丢弃路径率 | --drop-path 0.2
bn-momentum | BatchNorm动量 | --bn-momentum 0.1
bn-eps | BatchNorm epsilon | --bn-eps 1e-5
sync-bn | 同步 BatchNorm | --sync-bn
dist-bn | 在每个轮次结束后在节点之间分配 BatchNorm 统计数据 | --dist-bn reduce
split-bn | 每个增强分割使用单独的 BN 层 | --split-bn
model-ema | 启用模型指数移动平均 | --model-ema
model-ema-force-cpu | 强制模型EMA在CPU上跟踪 | --model-ema-force-cpu
model-ema-decay | 模型权重指数移动平均的衰减因子 | --model-ema-decay 0.999
model-ema-warmup | 启用模型EMA的热身 | --model-ema-warmup
seed | 随机种子 | --seed 42
worker-seeding | worker种子模式 | --worker-seeding all
log-interval | 训练状态日志间隔 | --log-interval 50
recovery-interval | 写入恢复检查点的批次间隔 | --recovery-interval 100
checkpoint-hist | 保留的检查点数量 | --checkpoint-hist 5
workers/-j | 训练进程数 | --workers 8
save-images | 每次日志记录保存输入批次的图像 | --save-images
pin-mem | 在DataLoader中固定CPU内存 | --pin-mem
no-prefetcher | 禁用快速预取器 | --no-prefetcher
output | 输出文件夹路径 | --output /path/to/output
experiment | 训练实验名称 | --experiment name
val-metric | 最佳指标 | --eval-metric top1 |
tta | 测试/推断时数据增强因子 | --tta 4 |
use-multi-epochs-loader | 使用多轮次加载器以节省时间 | --use-multi-epochs-loader |
log-wandb | 将训练和验证指标记录到wandb | --log-wandb |
