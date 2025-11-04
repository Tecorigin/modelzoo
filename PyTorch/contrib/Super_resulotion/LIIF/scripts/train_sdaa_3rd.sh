cd ..
pip install -r requirements.txt
pip install -e .

export SDAA_VISIBLE_DEVICES=0,1,2,3
timeout 2h bash ./tools/dist_train.sh configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py 4 | tee scripts/train_sdaa_3rd.log