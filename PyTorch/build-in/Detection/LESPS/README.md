# Mapping Degeneration Meets Label Evolution: Learning Infrared Small Target Detection with Single Point Supervision

**论文**：&nbsp;[**[Paper]**](https://arxiv.org/pdf/2304.01484.pdf) &nbsp; [**[Web]**](https://xinyiying.github.io/LESPS/) <br>

**该说明参考来源：https://github.com/XinyiYing/LESPS**

## 环境准备
- Python3.10
- Pytorch2.4.0
- Torch_sdaa2.0.0
- Ubuntu或Centos
- 使用一张太初T100加速卡训练
- 详细环境配置见`requirements.txt`

### 安装
拉取镜像
```bash
docker pull jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0
```
起docker
```bash
docker run -itd --name=<name> --net=host -v /mnt/:/mnt -v /mnt_qne00/:/mnt_qne00 -p 22 -p 8080 -p 8888 --privileged --device=/dev/tcaicard20 --device=/dev/tcaicard21 --device=/dev/tcaicard22 --device=/dev/tcaicard23 --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 300g jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.0.0-torch_sdaa2.0.0 /bin/bash
```

### 数据集准备
**SIRST3**：由作者整合的数据集.
下载地址： [Baidu Drive](https://pan.baidu.com/s/1NT2jdjS4wrliYYP0Rt4nXw?pwd=m6ui) .

* 数据集结构如下:
  ```
  ├──./datasets/SIRST3/
  │    ├── images
  │    │    ├── XDU0.png
  │    │    ├── Misc_1.png
  │    │    ├── ...
  │    │    ├── 001327.png
  │    ├── masks
  │    │    ├── XDU0.png
  │    │    ├── Misc_1.png
  │    │    ├── ...
  │    │    ├── 001327.png
  │    ├── masks_centroid
  │    │    ├── XDU0.png
  │    │    ├── Misc_1.png
  │    │    ├── ...
  │    │    ├── 001327.png
  │    ├── masks_coarse
  │    │    ├── XDU0.png
  │    │    ├── Misc_1.png
  │    │    ├── ...
  │    │    ├── 001327.png
  │    ├── img_idx
  │    │    ├── train_SIRST3.txt
  │    │    ├── test_SIRST3.txt  
  │    │    ├── test_SIRST.txt
  │    │    ├── test_NUDT-SIRST.txt
  │    │    ├── test_IRSTD-1K.txt
  ```
## 训练脚本
```bash
python train_full.py --model_names DNANet ALCNet ACM --dataset_names SIRST3 --label_type 'full'
```

## 训练输出
```bash
Apr 25 16:25:01 Epoch---1, total_loss---404.575378,
Apr 25 16:25:30 Epoch---2, total_loss---107.691422,
Apr 25 16:25:59 Epoch---3, total_loss---55.155720,
Apr 25 16:26:28 Epoch---4, total_loss---32.913757,
Apr 25 16:26:57 Epoch---5, total_loss---22.451338,
...
...
```
