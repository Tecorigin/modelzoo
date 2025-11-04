 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\shock_wave

 **4.数据集获取** 
无

 **5.训练代码** 

```
python shock_wave.py

```
 **6.测试代码** 

```
python shock_wave.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/shockwave/shock_wave_Ma2_pretrained.pdparams

```
 **7.结果评测** 
sdaa训练的loss为NA，cpu正常!
