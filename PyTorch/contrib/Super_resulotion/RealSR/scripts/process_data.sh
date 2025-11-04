cd ../codes
python3 ./preprocess/create_bicubic_dataset.py --dataset df2k --artifacts tdsr
python3 ./preprocess/collect_noise.py --dataset df2k --artifacts tdsr