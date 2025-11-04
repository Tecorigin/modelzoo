pip install timm==0.4.12
pip install opencv-python termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
cd ../kernels/window_process
export TORCH_SDAA_AUTOLOAD=cuda_migrate
python setup.py install
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path /mnt/nvme1/dataset/datasets/resnet50 --batch-size 64 > sdaa_log.log
