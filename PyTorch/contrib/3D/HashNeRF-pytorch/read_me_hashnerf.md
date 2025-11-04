创建数据结构
用load_data_by_kaggle.py下载nerf对应的数据


环境配置
conda create --name fairmot --clone torch_env

注释掉main函数中：torch.set_default_tensor_type('torch.sdaa.FloatTensor')
在：HashNeRF-pytorch/ray_utils.py", line 25,dir_bounds = directions.view(-1, 3)，把view换成reshape
在：HashNeRF-pytorch/utils.py", line 42，将c2w放到sdaa上：c2w = torch.FloatTensor(frame["transform_matrix"]).sdaa()
在：HashNeRF-pytorch/run_nerf.py", line 344，把torch.Tensor([1e10])放到sdaa上，dists = torch.cat([dists, torch.Tensor([1e10]).sdaa().expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
在：/HashNeRF-pytorch/run_nerf_helpers.py", line 12,把torch.log(torch.Tensor([10.]))放在sdaa上：mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).sdaa()
