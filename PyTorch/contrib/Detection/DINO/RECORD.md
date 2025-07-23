# DINO 迁移记录

## 1. 准备环境
```
conda create -n dino --clone torch_env
```
## 2. 下载代码
``` 
git clone https://github.geekery.cn/https://github.com/CoinCheung/BiSeNet.git
```
## 3. 安装依赖
本github仓库没有依赖列表。首先复制设备机器中的torch_env
再补充安装依赖。
```
conda activate dino
```
## 4. 修改头文件
vscode中ctrl+shift+F，将`import torch`替换为
```
import torch
import torch_sdaa

```
- 注意：末尾有一个回车符。
## 5. 修改训练代码
- vscode中ctrl+shift+F，将`cuda`替换为`sdaa`

## 6. 正常训练
修改数据集目录，打开`main_dino.py`121-123行
```python
    parser.add_argument('--data_path', default='/data/datasets/20241122/imagenet1k/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="log", type=str, help='Path to save logs and checkpoints.')
```

在终端运行下述代码进行单卡训练
```bash
export SDAA_VISIBLE_DEVICES=2
python -m torch.distributed.launch --nproc_per_node=1 --master_port 9292 --use_env main_dino.py --arch vit_small
```