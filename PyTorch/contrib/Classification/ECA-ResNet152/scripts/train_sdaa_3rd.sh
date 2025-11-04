python -m torch.distributed.launch --nproc_per_node=4 ../train.py --dataset_path /data/datasets/imagenet --batch_size 64 --epochs 3 --distributed True --lr 0.01 --autocast True
