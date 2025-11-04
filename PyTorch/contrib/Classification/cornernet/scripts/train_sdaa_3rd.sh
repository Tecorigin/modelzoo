cd ..
# pip install -r requirements.txt
torchrun tools/train.py configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py --launcher pytorch --nproc_per_node 4