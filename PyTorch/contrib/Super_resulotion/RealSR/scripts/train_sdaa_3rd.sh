cd ..
pip install -r requirements.txt
cd codes
export SDAA_VISIBLE_DEVICES=0,1,2,3
timeout 3h python -m torch.distributed.launch --nproc_per_node=4 train.py -opt options/df2k/train_bicubic_noise.yml \
    --launcher pytorch \
    | tee ../scripts/train_sdaa_3rd.log