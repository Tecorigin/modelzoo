cd ..
pip install -r requirements.txt
#单机单卡
python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet \
--batch_size 128 --epochs 10 --distributed True --lr 0.1 --autocast True
#单机单核组
# SDAA_LAUNCH_BLOCKING=1 python train.py --batch_size 128 --epochs 10 --distributed False --dataset_path /data/datasets/imagenet
#  --lr 0.1 --autocast True
