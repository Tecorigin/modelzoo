cd ../
export SDAA_VISIBLE_DEVICES=0,1,2,3
export TORCH_HOME=/data/datasets/checkpoints/
timeout 120m python -m torch.distributed.launch --nproc_per_node=4 --master_port 9292 --use_env main_dino.py