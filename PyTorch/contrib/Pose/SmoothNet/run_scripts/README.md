### 参数介绍

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
model_name |模型名称。 | --model_name smoothnet
epoch | 训练轮数。 | --epoch 10
batch_size/bs | batch_size。 |  --batch_size 4000 / --bs 4000
device | 用于训练或推理的设备。 | --device sdaa / --device cpu
eval | 是否进行验证。 | --eval
autocast | 是否开启AMP训练。 | --autocast
cfg | 配置文件路径。 | --cfg configs/h36m_fcn_3D.yaml
dataset_name | 数据集名称。 | --dataset_name h36m
estimator | 姿态估计器名称。 | --estimator fcn
body_representation | 姿态表示类型。 | --body_representation 3D
slide_window_size | 滑动窗口大小。 | --slide_window_size 32
checkpoint | 验证时的权重路径。 | --checkpoint results/path_to/epoch_23.pth.tar
