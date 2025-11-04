#克隆torch_sdaa库，激活环境并安装matplotlib
#cd到指定路径下，源码是一个比较大的框架（根据实际情况）
cd ../references/classification
#ps:如果有遗漏，pip安装即可
pip install -r requirements.txt

#运行指令
#注意此时使用四卡，且指定了模型名称为Wide_ResNet101_2以及数据集的地址（数据集里面需要包括train和val文件）
 
torchrun --nproc_per_node=4 train.py\
     --model regnet_x_400mf --epochs 100 --batch-size 128 --wd 0.00005 --lr=0.8\
     --lr-scheduler=cosineannealinglr --lr-warmup-method=linear\
     --lr-warmup-epochs=5 --lr-warmup-decay=0.1