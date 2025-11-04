cd ..
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
SDAA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./deep_sort/deep/train_multiGPU.py --data-dir /data/datasets/Market-1501-v15.09.15 --weights  ./deep_sort/deep/checkpoint/ckpt.t7 
#python ./deep_sort/deep/train.py --data-dir /data/datasets/Market-1501-v15.09.15 --weights ./deep_sort/deep/checkpoint/ckpt.t7