 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\tempoGAN

 **4.数据集获取** 


```
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat -P datasets/tempoGAN/
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/
```


 **5.训练代码** 

```
python tempoGAN.py

```
 **6.测试代码** 

```
python biharmonic2d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/biharmonic2d/biharmonic2d_pretrained.pdparams

```
 **7.PIP 安装** 

```
pip install hdf5storage
```

 **8.结果评测** 
sdaa上无法运行，cpu上可以
![输入图片说明](https://foruda.gitee.com/images/1742527180327482015/83226ad8_12173785.png "090b6c5641ea277751099bc2dd2375c.png")