#!/bin/bash
cd ..


torchrun --nproc_per_node 4 run_nerf.py --training_time 2 --config configs/fern.txt --use_amp=True --use_DDP=True

#单卡训练：python run_nerf.py --config configs/fern.txt