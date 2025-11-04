pip install -r requirements.txt

./distributed_train.sh 4 /mnt/dataset/imagenet --model seresnet50 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel  --log-interval 1 -j 4 > seresnet50.log 2>&1

cd scripts
python loss_check.py 