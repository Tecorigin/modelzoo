**1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\amgnet

 **4.数据集获取** 

```
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip
# windows
# curl https://paddle-org.bj.bcebos.com/paddlescience/datasets/AMGNet/data.zip -o data.zip
# unzip it
unzip data.zip
```
 **5.安装 Paddle Graph Learning 图学习工具和 PyAMG 代数多重网格工具** 

```
pip install -r requirements.txt
```


 **6.训练代码** 

```
python amgnet_airfoil.py

```
 **6.cpu训练代码** 

```
python amgnet_airfoil_cpu.py

```
 **7.测试代码** 

```
python amgnet_airfoil.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/amgnet/amgnet_airfoil_pretrained.pdparams

```
 **8.结果评测** 
sdaa上算子不支持，fall back 到cpu上可正常训练
![输入图片说明](https://foruda.gitee.com/images/1738909436204913587/dbe0f3e8_12173785.png "b9ebd091bab71ef7e25a1c143bec9c6.png")