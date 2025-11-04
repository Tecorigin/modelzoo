#!/bin/bash
pip install -r requirements.txt

python -m torch.distributed.launch --nproc_per_node=4  tools/train.py -f configs/damoyolo_tinynasL25_S.py > damoyolo_tinynasL25_S.log 2>&1 &

cd scripts
python loss_check.py