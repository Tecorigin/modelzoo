 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\spinn

 **4.数据集获取** 

无

 **5.训练代码** 

```
python helmholtz3d.py

```
 **6.测试代码** 

```
python helmholtz3d.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/spinn/spinn_helmholtz3d_pretrained.pdparams

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势，具体信息可见train——log.TXT
![输入图片说明](https://foruda.gitee.com/images/1740134240328343544/04578de5_12173785.png "06cf594d8128a3201565ad5e3e7c019.png")