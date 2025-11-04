
cd ..    # 进入上一级目录
pip install -r requirements.txt
cd src   # 进入src文件夹
#单机单卡
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 --use_env main.py  --dataset imagenet --dataset_path /data/datasets/imagenet
# 单机单核组
# python main.py --no_distributed  --dataset imagenet --dataset_path /data/datasets/imagenet