 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\laplace

 **4.数据集获取** 

无

 **5.训练代码** 

```
python laplace2d.py

```
 **6.测试代码** 

```
python laplace2d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/laplace2d/laplace2d_pretrained.pdparams

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势，loss：477-》19
![输入图片说明](https://foruda.gitee.com/images/1738902477374422915/dfa924f3_12173785.png "3a2f28df18e842b7f22f8df7438a37e.png")