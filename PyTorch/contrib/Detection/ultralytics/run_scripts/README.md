## 运行样例

#### 模型训练
* 单卡四核组为例：

    ```bash
    # 检测模型训练
    python run_scripts/run_train_yolo.py --mode=detect --data=coco8.yaml --model=yolov8n.pt --epochs=100 --imgsz=640 --device=[0,1,2,3] --batch=64

    # 分割模型训练
    python run_scripts/run_train_yolo.py --mode=segment --data=coco8-seg.yaml --model=yolov8n-seg.pt --epochs=100 --imgsz=640 --device=[0,1,2,3] --batch=64

    # 分类模型训练
    python run_scripts/run_train_yolo.py --mode=classify --data=mnist160 --model=yolov8n-cls.pt --epochs=100 --imgsz=640 --device=[0,1,2,3] --batch=64

    # 姿态模型训练
    python run_scripts/run_train_yolo.py --mode=pose --data=coco8-pose.yaml --model=yolov8n-pose.pt --epochs=100 --imgsz=640 --device=[0,1,2,3] --batch=64

    # OBB 模型训练
    python run_scripts/run_train_yolo.py --mode=obb --data=dota8.yaml --model=yolov8n-obb.pt --epochs=100 --imgsz=640 --device=[0,1,2,3] --batch=64
    ```

### 参数介绍
参数名 | 解释 | 样例 
-----------------|-----------------|----------------- 
mode|模型训练模式，支持 detect, segment, classify, pose, obb| --mode detect
model|指定用于训练的模型文件。接受 .pt 预训练模型或 .yaml 配置文件的路径。对于定义模型结构或初始化权重是必不可少的。|--model example_model.pt
data|数据集配置文件的路径（例如 coco8.yaml）。该文件包含数据集特定的参数，包括训练和验证数据的路径、类别名称和类别数量。|--data coco8.yaml
epochs|总训练周期数。每个周期表示对整个数据集的完整遍历。调整此值会影响训练时间和模型性能。|--epochs 100
time|最长训练时间（小时）。如果设置，将覆盖 epochs 参数，允许在指定时间后自动停止训练。对于时间受限的训练场景很有用。|--time 5.0
patience|在验证指标没有改善的情况下等待的周期数，然后提前停止训练。通过在性能达到平稳时停止训练来防止过拟合。|--patience 100
batch|训练批次大小，表示在模型内部参数更新之前处理的图像数量。AutoBatch（batch=-1）根据 GPU 内存可用性动态调整批次大小。|--batch 16
imgsz|训练的目标图像大小。所有图像在输入模型前都调整为此尺寸。影响模型准确性和计算复杂度。|--imgsz 640
save|启用训练检查点和最终模型权重的保存。对于恢复训练或模型部署很有用。|--save True
save_period|模型检查点保存频率（以周期为单位）。设置为 -1 可禁用此功能。在长时间训练期间保存中间模型很有用。|--save_period -1
cache|启用将数据集图像缓存到内存（True/ram）、磁盘（disk）或禁用（False）。通过减少磁盘 I/O 提高训练速度，但会增加内存使用。|--cache False
device|指定训练的计算设备：单个 GPU（device=0）、多个 GPU（device=0,1）、CPU（device=cpu）或 Apple 硅设备（device=mps）。|--device 0
workers|数据加载的工作线程数（每个 RANK，如果是多 GPU 训练）。影响数据预处理和输入模型的速度，特别是在多 GPU 设置中有用。|--workers 8
project|保存训练输出的项目目录名称。允许有条理地存储不同的实验。|--project my_project
name|训练运行的名称。用于在项目文件夹中创建子目录，存储训练日志和输出。|--name my_training_run
exist_ok|如果为 True，则允许覆盖现有的项目/名称目录。对于无需手动清除以前输出的迭代实验很有用。|--exist_ok False
pretrained|确定是否从预训练模型开始训练。可以是布尔值或加载权重的特定模型路径。提高训练效率和模型性能。|--pretrained True
optimizer|训练时的优化器选择。选项包括 SGD、Adam、AdamW、NAdam、RAdam、RMSProp 等，或根据模型配置自动选择。影响收敛速度和稳定性。|--optimizer auto
verbose|启用训练期间的详细输出，提供详细的日志和进度更新。对于调试和密切监视训练过程很有用。|--verbose False
seed|设置训练的随机种子，确保在相同配置下结果的可重复性。|--seed 0
deterministic|强制使用确定性算法，确保可重复性，但可能会因为限制非确定性算法而影响性能和速度。|--deterministic True
single_cls|在训练期间将多类数据集中的所有类视为一个类。对于二分类任务或关注对象存在而非分类时有用。|--single_cls False
rect|启用矩形训练，优化批次组合以最小化填充。可以提高效率和速度，但可能会影响模型准确性。|--rect False
cos_lr|使用余弦学习率调度器，按照余弦曲线调整学习率。帮助更好地管理学习率收敛。|--cos_lr False
close_mosaic|在最后 N 个周期中禁用马赛克数据增强，以在完成前稳定训练。设置为 0 禁用此功能。|--close_mosaic 10
resume|从上次保存的检查点恢复训练。自动加载模型权重、优化器状态和周期数，无缝继续训练。|--resume False
amp|启用自动混合精度（AMP）训练，减少内存使用并可能加速训练，对准确性影响最小。|--amp True
fraction|指定用于训练的数据集比例。允许在资源有限时使用数据集的子集进行训练，或用于实验。|--fraction 1.0
profile|启用训练期间的 ONNX 和 TensorRT 速度分析，对于优化模型部署很有用。|--profile False
freeze|冻结模型的前 N 层或按索引指定的层，减少可训练参数数量。对于微调或迁移学习有用。|--freeze None
lr0|初始学习率（例如 SGD=1E-2，Adam=1E-3）。调整此值对优化过程至关重要，影响模型权重的更新速度。|--lr0 0.01
lrf|最终学习率是初始率的一个分数 = (lr0 * lrf)，与调度器一起使用以随时间调整学习率。|--lrf 0.01
momentum|SGD 的动量因子或 Adam 优化器的 beta1，影响当前更新中纳入过去梯度的程度。|--momentum 0.937
weight_decay|L2 正则化项，惩罚较大的权重以防止过拟合。|--weight_decay 0.0005
warmup_epochs|学习率预热的周期数，逐渐将学习率从低值增加到初始学习率，以在初期稳定训练。|--warmup_epochs 3.0
warmup_momentum|预热阶段的初始动量，在预热期内逐渐调整到设置的动量。|--warmup_momentum 0.8
warmup_bias_lr|预热阶段偏置参数的学习率，有助于在初期周期内稳定模型训练。|--warmup_bias_lr 0.1
box|损失函数中框损失组件的权重，影响准确预测边界框坐标的重视程度。|--box 7.5
cls|总损失函数中分类损失的权重，影响正确类别预测相对于其他组件的重要性。|--cls 0.5
dfl|分布焦点损失的权重，在某些 YOLO 版本中用于细粒度分类。|--dfl 1.5
pose|在姿态估计训练中姿态损失的权重，影响准确预测姿态关键点的重视程度。|--pose 12.0
kobj|姿态估计模型中的关键点对象损失的权重，平衡检测置信度与姿态准确性。|--kobj 2.0
label_smoothing|应用标签平滑，将硬标签软化为目标标签和标签的均匀分布的混合物，可以提高泛化能力。|--label_smoothing 0.0
nbs|用于损失归一化的名义批次大小。|--nbs 64
overlap_mask|确定在训练期间分割掩码是否应重叠，适用于实例分割任务。|--overlap_mask True
mask_ratio|分割掩码的下采样比例，影响训练期间使用的掩码分辨率。|--mask_ratio 4
dropout|分类任务中的 dropout 率，通过在训练期间随机省略单元来防止过拟合。|--dropout 0.0
val|启用训练期间的验证，允许定期评估模型在单独数据集上的性能。|--val True
plots|生成并保存训练和验证指标以及预测示例的图表，提供模型性能和学习进展的视觉洞察。|--plots False
hsv_h|按色轮的一部分调整图像的色调，引入颜色变化。帮助模型在不同照明条件下进行泛化。|--hsv_h 0.015
hsv_s|按比例调整图像的饱和度，影响颜色的强度。对于模拟不同的环境条件很有用。|--hsv_s 0.7
hsv_v|按比例修改图像的亮度，帮助模型在各种光照条件下表现良好。|--hsv_v 0.4
degrees|在指定的度数范围内随机旋转图像（-180 到 +180），提高模型识别各种方向的物体的能力。|--degrees 0.0
translate|按图像大小的一部分水平和垂直平移图像，帮助学习检测部分可见的物体。|--translate 0.1
scale|按增益因子（>=0.0）缩放图像，模拟不同距离摄像机的物体。|--scale 0.5
shear|按指定的度数范围剪切图像（-180 到 +180），模拟从不同角度查看的效果。|--shear 0.0
perspective|对图像应用随机透视变换（0.0 - 0.001），增强模型理解 3D 空间中的物体的能力。|--perspective 0.0
flipud|以指定概率（0.0 - 1.0）上下翻转图像，在不影响物体特性的情况下增加数据变化。|--flipud 0.0
fliplr|以指定概率（0.0 - 1.0）左右翻转图像，对于学习对称物体和增加数据集多样性很有用。|--fliplr 0.5
bgr|以指定概率（0.0 - 1.0）将图像通道从 RGB 翻转为 BGR，对于增加对错误通道顺序的鲁棒性很有用。|--bgr 0.0
mosaic|将四张训练图像组合成一张（0.0 - 1.0），模拟不同场景组成和物体交互。对于复杂场景理解非常有效。|--mosaic 1.0
mixup|混合两张图像及其标签（0.0 - 1.0），创建复合图像。通过引入标签噪声和视觉变化增强模型的泛化能力。|--mixup 0.0
copy_paste|将一个图像中的物体复制并粘贴到另一个图像上（0.0 - 1.0），有助于增加物体实例并学习物体遮挡。|--copy_paste 0.0
auto_augment|自动应用预定义的增强策略（randaugment, autoaugment, augmix），通过多样化视觉特征优化分类任务。|--auto_augment randaugment
erasing|在分类训练期间随机擦除图像的一部分（0.0 - 0.9），鼓励模型关注不太明显的特征进行识别。|--erasing 0.4
crop_fraction|将分类图像裁剪为其大小的一部分（0.1 - 1.0），以强调中央特征并适应物体比例，减少背景干扰。|--crop_fraction 1.0
