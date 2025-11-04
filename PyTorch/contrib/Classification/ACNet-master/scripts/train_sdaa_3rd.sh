#进入当前目录
cd ..

# 克隆torch_sdaa环境，激活环境
#此外，需要额外下载如下库
# pip install h5py
# pip install tensorboard
# pip install coloredlogs
# pip install matplotlib

pip install -r requirements.txt

#Make a soft link to your ImageNet directory, which contains "train" and "val" directories.
#ln -s YOUR_PATH_TO_IMAGENET imagenet_data
ln -s /data/datasets/imagenet imagenet_data

#Set the environment variables. We use 4 GPUs with Distributed Data Parallel. 
export PYTHONPATH=.
export SDAA_VISIBLE_DEVICES=0,1,2,3

#Train a ResNet-18 on ImageNet with Asymmetric Convolution Blocks. 
python -m torch.distributed.launch --nproc_per_node=4 acnet/do_acnet.py -a sres18 -b acb 