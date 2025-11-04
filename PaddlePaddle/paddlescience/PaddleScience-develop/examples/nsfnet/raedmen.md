 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\nsfnet

 **4.数据集获取** 

```
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/NSF4_data.zip -P ./data/
unzip ./data/NSF4_data.zip
```

 **5.训练代码** 

```
python VP_NSFNet4.py data_dir=./data/

```
 **6.测试代码** 

```
python VP_NSFNet4.py    mode=eval  data_dir=./data/  EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet4.pdparams

```
 **7.结果评测** 
sdaa上无法训练，fallback cpu可以正常训练：
![输入图片说明](https://foruda.gitee.com/images/1739004648836408018/7559abc9_12173785.png "fe59ec9c798af2ce7bdf2458cb425ba.png")