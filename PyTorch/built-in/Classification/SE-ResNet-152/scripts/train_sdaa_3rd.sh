pip install -r requirements.txt

./distributed_train.sh 4 /mnt_qne00/dataset/imagenet --model seresnet152  --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --log-interval 1 -j 4 > seresnet152.log 2>&1

cd scripts
python loss_check.py 