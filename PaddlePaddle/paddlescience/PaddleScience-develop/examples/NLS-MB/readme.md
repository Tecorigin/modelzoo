 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\NLS-MB

 **4.数据集获取** 
无

 **5.训练代码** 

```
# soliton
python NLS-MB_optical_soliton.py
# rogue wave
python NLS-MB_optical_rogue_wave.py

```
 **5.训练代码_CPU** 

```
# soliton
python NLS-MB_optical_soliton_cpu.py
# rogue wave
python NLS-MB_optical_rogue_wave_cpu.py

```
 **6.测试代码** 

```
# soliton
python NLS-MB_optical_soliton.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_soliton_pretrained.pdparams
# rogue wave
python NLS-MB_optical_rogue_wave.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/NLS-MB/NLS-MB_rogue_wave_pretrained.pdparams

```
 **7.结果评测** 
sdaa上算子不支持，fall back cpu 可正常训练，错误如下：
![输入图片说明](https://foruda.gitee.com/images/1738908222162633301/0858282e_12173785.png "52a2deae8381db22ecb9341b2c09b83.png")

不支持的算子为：
 File "/root/miniconda3/envs/paddle_env/lib/python3.10/site-packages/paddle/tensor/math.py", line 518, in pow
    return _C_ops.elementwise_pow(x, y)
![输入图片说明](https://foruda.gitee.com/images/1739956854378578264/7537d059_12173785.png "屏幕截图")