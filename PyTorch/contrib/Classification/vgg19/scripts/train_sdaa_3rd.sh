cd ..
pip install -r requirements.txt

# torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a vgg19 -b 256
# torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29503 main.py /data/datasets/imagenet -a vgg19 -b 256 >> ./scripts/train_sdaa_3rd_vgg19.log 2>&1
torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29503 main.py /data/datasets/imagenet -a vgg19 -b 256