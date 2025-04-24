<img src="./images/denoising-diffusion.png" width="500px"></img>

来源官方文档，建议参考官方文档：https://github.com/lucidrains/denoising-diffusion-pytorch

## Denoising Diffusion Probabilistic Model, in Pytorch

基于PyTorch实现的去噪扩散概率模型（Denoising Diffusion Probabilistic Model）是一种生成建模的新方法，具有与生成对抗网络（GANs）竞争的潜力。它通过去噪得分匹配（Denoising Score Matching）估算数据分布的梯度，随后通过Langevin采样从真实分布中进行采样。

## 文件结构
```bash
├── coverage.py                                                 #查看覆盖率
├── deduplicate_cases_and_save.py                               #筛选算子用的
├── denoising_diffusion_pytorch                                 #框架实现目录
│   ├── attend.py
│   ├── classifier_free_guidance.py
│   ├── continuous_time_gaussian_diffusion.py
│   ├── denoising_diffusion_pytorch_1d.py
│   ├── denoising_diffusion_pytorch.py
│   ├── elucidated_diffusion.py
│   ├── fid_evaluation.py
│   ├── guided_diffusion.py
│   ├── __init__.py
│   ├── karras_unet_1d.py
│   ├── karras_unet_3d.py
│   ├── karras_unet.py
│   ├── learned_gaussian_diffusion.py
│   ├── repaint.py
│   ├── simple_diffusion.py
│   ├── version.py
│   ├── v_param_continuous_time_gaussian_diffusion.py
│   └── weighted_objective_gaussian_diffusion.py
├── dump_info_denoising_diffusion.tar         
├── images
│   ├── denoising-diffusion.png                  
│   ├── loss.jpg
│   └── sample.png
├── LICENSE
├── README.md
├── requirements.txt
├── scripts                                         #训练脚本
│   └── train.py
├── setup.py
├── training_cuda.og
└── training_sdaa.log
```

## 使用说明

### Trainng

#### 快速开始
* 获取数据集
选择图像数量大于100的任意图像数据集，这里我选择idenprof中doctor数据集
```
存放路径： /mnt/nvme1/dataset/datasets/idenprof
```
* 起docker环境

使用该镜像
```
docker pull jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0
```
创建环境
```
docker run -itd --name=<name> --net=host -v /data/application/hongzg:/data/application/hongzg -v /mnt/:/mnt -v /hpe_share/:/hpe_share -p 22 -p 8080 -p 8888 --privileged --device=/dev/tcaicard20 --device=/dev/tcaicard21 --device=/dev/tcaicard22 --device=/dev/tcaicard23 --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 300g jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0 /bin/bash
```
其他依赖库 参考requirements.txt
```
--pytorch-fid
--einops
--accelerate
--ema-pytorch
```
* 训练代码：
可以根据官方文档在Trainer中修改参数
```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch_sdaa.utils import cuda_migrate

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/mnt/nvme1/application/hongzg/dataset/idenprof/train/doctor',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 1,              # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()
```
* 设置完成参数后，在仓库根目录下执行**python -m ./scripts.train**即可开始训练
