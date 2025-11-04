cd ../
export SDAA_VISIBLE_DEVICES=0,1,2,3
export TORCH_HOME=/data/datasets/checkpoints/
timeout 120m python3 -m torch.distributed.launch --use-env --nproc_per_node=4 main.py recognition -c config/st_gcn/kinetics-skeleton/train.yaml | tee ./scripts/train_sdaa_3rd.log