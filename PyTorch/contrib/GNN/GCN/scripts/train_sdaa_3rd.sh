cd ../examples/pytorch/gcn/

torchrun --nproc_per_node 4 train.py --dataset cora --use_amp True

