 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\heat_exchanger

 **4.数据集获取** 

无

 **5.训练代码** 

```
python heat_exchanger.py

```
 **5.训练代码（cpu）** 

```
python heat_exchanger_cpu.py

```
 **6.测试代码** 

```
python heat_exchanger.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/HEDeepONet/HEDeepONet_pretrained.pdparams

```
 **7.结果评测** 
sdaa上无法进行训练，fall back cpu可以正常训练：
![输入图片说明](https://foruda.gitee.com/images/1739006787079279849/4fba37b1_12173785.png "106cd5ab05a8badf37a5f75a46bab77.png")
不支持算子信息：
 File "/root/miniconda3/envs/paddle_env/lib/python3.10/site-packages/paddle/tensor/math.py", line 518, in pow
    return _C_ops.elementwise_pow(x, y)
ValueError: (InvalidArgument) The input tensor x and y dims must be same in ElementwisePowRawKernel
  [Hint: Expected x.dims() == y.dims(), but received x.dims():1000, 1 != y.dims():.] (at /data/qne/ci_env/teco_ap/workspace/teco-paddle_sdaa-build/usertestdir_4/backends/sdaa/kernels/elementwise_pow_kernel.cc:42)
![输入图片说明](https://foruda.gitee.com/images/1739958181047068983/027f9abb_12173785.png "屏幕截图")