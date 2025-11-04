cd ..
pip install -r requirements.txt
python main.py /data/datasets/imagenet --gpu 0 --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:65501' --world-size 1 --rank 0 --dist-backend 'tccl' --workers 40