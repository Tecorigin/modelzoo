cd ..

MASTER_PORT=29501 python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 29501 train.py \
    --dataset_path /root/data/bupt/datasets/imagenet \
    --batch_size 32 --epochs 10 --distributed True --lr 0.0001 --autocast True

