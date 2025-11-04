 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\adv

 **4.数据集获取** 


运行模型前请在 Zhengyu-Huang/Operator-Learning 中下载 adv_a0.npy 和 adv_aT.npy 两个文件，并将其放在 ./examples/adv/data/ 文件夹下。
Zhengyu-Huang/Operator-Learning的地址为：https://github.com/Zhengyu-Huang/Operator-Learning/tree/main/data

 **5.训练代码** 

```
python adv_cvit.py
```
 **5.训练代码（cpu）** 

```
# tfno 模型训练
python adv_cvit_cpu.py

```
 **6.测试代码** 

```
python adv_cvit.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cvit/adv_cvit_pretrained.pdparams

```
 **7.结果评测** 
sdaa上无法运行，fall back cpu上正常训练
![输入图片说明](https://foruda.gitee.com/images/1742525058029605946/c758e3c1_12173785.png "1b217b3cdba3cf06b1912d0653cf71e.png")

 **7.不支持算子信息**
![输入图片说明](https://foruda.gitee.com/images/1742525058029605946/c758e3c1_12173785.png "1b217b3cdba3cf06b1912d0653cf71e.png")