## 模型介绍

目前支持的模型有`resnet50`, `resnet18`, `resnet101`, `resnet50_v10`, `fused_resnet50`。
1. `resnet50`和`resnet50_v10`分别指的是resnet50的两个不同版本，v1.5和v1.0。ResNet v1.0为ResNet论文中所使用的版本，ResNet50 v1.5是ResNet50 v1.0的修改版本。主要区别在于bottleneck块中的下采样操作。在v1.0版本中，第一个1x1卷积层的步长（stride）为2，而在v1.5版本中，3x3卷积层的步长为2。这种差异使得ResNet50 v1.5的精度略高于v1.0版本，约高出0.5%，但在图像处理速度上略有下降，约降低5%。实际使用中多为v1.5版本。
2. `resnet18`与`resnet101`为与resnet50的同系列模型
3. `fused_resnet50`为自研实现的融合版本resnet50，将所有算子融合为一个大算子，运行速度更快，且不影响精度。