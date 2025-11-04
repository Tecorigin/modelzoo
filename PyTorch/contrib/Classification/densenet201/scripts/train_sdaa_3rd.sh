cd ..
pip install -r requirements.txt
# torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a densenet201 -b 64
# torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29503 main.py /data/datasets/imagenet -a densenet201 -b 64 >> ./scripts/train_sdaa_3rd_densenet201.log 2>&1
torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29505 main.py /data/datasets/imagenet -a densenet201 -b 64