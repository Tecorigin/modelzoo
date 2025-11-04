 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\brusselator3d

 **4.数据集获取** 

```
wget -P data -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/Brusselator3D/brusselator3d_dataset.npz
```

 **5.训练代码** 

```
python brusselator3d.py
```
 **5.训练代码（cpu）** 

```
# tfno 模型训练
python brusselator3d_cpu.py

```
 **6.测试代码** 

```
python brusselator3d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/Brusselator3D/brusselator3d_pretrained.pdparams

```
 **7.结果评测** 
sdaa上无法运行，fall back cpu上正常训练
![输入图片说明](https://foruda.gitee.com/images/1742523725313278641/2cd7d9fd_12173785.png "ab567ed2f59a865321c8016c1ad4e77.png")

 **7.不支持算子信息**
![输入图片说明](https://foruda.gitee.com/images/1742523725313278641/2cd7d9fd_12173785.png "ab567ed2f59a865321c8016c1ad4e77.png")