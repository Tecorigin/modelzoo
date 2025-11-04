cd ../references/detection
pip install -r requirements.txt


#开始运行代码
torchrun --nproc_per_node=4 train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1