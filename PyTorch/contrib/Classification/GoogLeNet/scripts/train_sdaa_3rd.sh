cd ..

python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /root/data/bupt/datasets/imagenet \
--batch_size 32 --epochs 10 --distributed True --lr 0.0001 --autocast True