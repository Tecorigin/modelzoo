cd ..
pip install -r requirements.txt
#单机单卡
python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet \
--batch_size 64 --epochs 1 --distributed True --lr 0.00015 --autocast True
#单机单核组
#SDAA_LAUNCH_BLOCKING=1 python train.py --batch_size 64 --epochs 1 --distributed False --dataset_path /data/datasets/imagenet
#--lr 0.00015 --autocast True
