# 1.模型概述
NeRF（Neural Radiance Fields）是一种基于神经网络的方法，用于从一组输入图像中学习并表示复杂场景的3D结构和外观。NeRF模型通过预测空间中每一点沿各个方向的光线密度和颜色来重建场景，能够生成高质量的新视角合成图像。这种方法不仅有效地捕捉了场景的几何形状，而且也精细地模拟了光照效果，从而实现了逼真的3D场景再现。NeRF模型在计算机视觉与图形学领域引起了广泛关注，特别是在虚拟现实、增强现实以及3D场景重建等方面展示了巨大潜力。当前支持的NeRF模型变种包括：原始NeRF、Mip-NeRF（改进版本，提高了渲染效率和质量）、以及IBRNet（一种结合图像特征匹配与NeRF方法的混合模型）。NeRF及其衍生模型代表了现代3D场景表示和生成技术的重要进展。

# 2.快速开始
使用本模型执行训练的主要流程如下：
    基础环境安装：介绍训练前需要完成的基础环境检查和安装。
    获取数据集：介绍如何获取训练所需的数据集。
    构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
    启动训练：介绍如何运行训练。
2.1 基础环境安装
    从已有的torch_env拷贝环境，设虚拟环境名称为nerf
    conda create --name nerf --clone torch_env
    安装相关库函数：
    conda install --file conda_requirements.txt

2.2 数据集获取
    安装kaggle相关库（2.1安装过这里就无需安装）
    pip install kagglehub
    下载数据集：运行load_data_by_kaggle.py，下载的数据集会存放在/root/.cache/kagglehub/datasets/arenagrenade/llff-dataset-full/versions/1
    将数据集拷贝到指定目录下：cp -r 原路径（上述下载路径） /nerf-pytorch-master/data/
    数据集目录结构：
    |----data
           |----nerf_llff_data
                    |------fern
                            |-----images
                            |-----images_4
                            |-----images_8
                            |----sparse
                            |----......
                    |------flower
                            |----......
                    |------......

2.3 启动训练
    支持单卡、多卡以及混合精度训练（amp）
    单卡训练：python run_nerf.py --config configs/fern.txt
    多卡混合精度训练：torchrun --nproc_per_node 4 run_nerf.py --config configs/fern.txt --use_amp=True --use_DDP=True
    注:1.如果不采用混合精度训练，就去掉上述指令的--use_amp=True
       2.切换数据集时，在--config configs/fern.txt中替换指定数据集txt文件即可

2.5训练结果
训练时长预计41h，选取前2个小时训练结果
|------|------|
|loss  |PSNR  |
|------|------|
|0.667 |4.767 |
|------|------|
模型训练结果保存在script/logs/fern_test/目录下
绘制loss以及PSNR曲线，发现loss没有明显下降，PSNR没有明显上升，分析原因：由于nerf训练3D数据难度大，耗时久，因此前期训练（2h）loss不会有太大的下降趋势。


# 3.修改部分
1.torch.set_default_tensor_type() 已弃用：
根据警告信息，torch.set_default_tensor_type() 在 PyTorch 2.1 版本中已被弃用。官方建议使用 torch.set_default_dtype()

2.把load_llff.py文件的第110行的ignoregamma=True去掉

3.run_nerf.py文件358行的t_vals放到sdaa上：t_vals = torch.linspace(0., 1., steps=N_samples).sdaa()

4.run_nerf_helpers.py文件第11行，torch.Tensor([10.])放到sdaa上：mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).sdaa())

5.run_nerf.py的279行，将torch.Tensor([1e10])放到sdaa上：dists = torch.cat([dists, torch.Tensor([1e10]).sdaa().expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

