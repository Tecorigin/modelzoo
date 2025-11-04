cd ..
pip install -r requirements.txt
pip install -e .
export SDAA_VISIBLE_DEVICES=0,1,2,3
timeout 3h bash ./tools/dist_train.sh configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py 4 | tee scripts/train_sdaa_3rd.log