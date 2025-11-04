#克隆torch_sdaa库，激活环境并安装matplotlib
#cd到指定路径下，源码是一个比较大的框架（根据实际情况）
cd ../references/classification
#ps:如果有遗漏，pip安装即可
pip install -r requirements.txt

#运行指令
#注意此时使用四卡，且指定了模型名称为Wide_ResNet101_2以及数据集的地址（数据集里面需要包括train和val文件）

torchrun --nproc_per_node=4 train.py\
 --model swin_v2_t --epochs 300 --batch-size 64 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 256 --val-crop-size 256 --train-crop-size 256 

