cd ..
pip install -r requirements.txt
cd v0.5.0/nvidia/submission/code/recommendation/pytorch
export SDAA_VISIBLE_DEVICES=0,1,2,3
timeout 2h python -m torch.distributed.launch --nproc_per_node 4 ncf.py /data/datasets/ml-20m -b 8192 --AMP