cd ..
pip install -r requirements.txt
pip install -e .

# Train the model
mkdir -p checkpoints/fconv_wmt_en_de
esport SDAA_VISIBLE_DEVICES=0,1,2,3 
timeout 2h fairseq-train \
    /data/datasets/wmt14_en_de \
    --arch fconv_wmt_en_de \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000 \
    --save-dir checkpoints/fconv_wmt_en_de \
    --log-format json \
    --log-interval 2 \
    --log-file scripts/train_sdaa_3rd.log \
    --amp \

# Evaluate
# fairseq-generate data-bin/wmt14_en_de \
#     --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
#     --beam 5 --remove-bpe