cd ..
pip install -r requirements.txt

# distributed True for nproc_per_node 4
# Adapted to tecorigin hardware
#单机单卡
python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet \
--batch_size 256 --epochs 11 --distributed True --lr 0.5 --autocast True

#单机单核组
# python train.py --batch_size 128 --epochs 11 --distributed False 
#--dataset_path /data/datasets/imagenet--lr 0.04 --autocast True


