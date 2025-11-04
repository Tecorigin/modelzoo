**1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\nsfnet

 **4.数据集获取** 

```
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/NSFNet/cylinder_nektar_wake.mat -P ./data/
```

 **5.训练代码** 

```
# VP_NSFNet1
python VP_NSFNet1.py
# VP_NSFNet2
python VP_NSFNet2.py data_dir=./data/cylinder_nektar_wake.mat
# VP_NSFNet3
python VP_NSFNet3.py
```

 **5.训练代码（cpu）** 

```
# VP_NSFNet2
python VP_NSFNet2_cpu.py data_dir=./data/cylinder_nektar_wake.mat
# VP_NSFNet3
python VP_NSFNet3_cpu.py
```
 **6.测试代码** 

```
# VP_NSFNet1
python VP_NSFNet1.py    mode=eval  pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet1.pdparams

# VP_NSFNet2

python VP_NSFNet2.py    mode=eval  data_dir=./data/cylinder_nektar_wake.mat  pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet2.pdparams

# VP_NSFNet3
python VP_NSFNet3.py    mode=eval  pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/nsfnet/nsfnet3.pdparams

```
 **7.结果评测** 
VP_NSFNet1设备成功转移到sdaa，VP_NSFNet2，VP_NSFNet3无法在sdaa上进行训练fallback cpu可以进行：
![输入图片说明](https://foruda.gitee.com/images/1739003429043939618/1a04f72a_12173785.png "05b75b67cbd50c1b4567590684b7f43.png")

不支持算子信息：
 File "/root/miniconda3/envs/paddle_env/lib/python3.10/site-packages/paddle/base/dygraph/base.py", line 830, in grad
    return core.eager.run_partial_grad(
![输入图片说明](https://foruda.gitee.com/images/1739957846847327021/53418170_12173785.png "屏幕截图")