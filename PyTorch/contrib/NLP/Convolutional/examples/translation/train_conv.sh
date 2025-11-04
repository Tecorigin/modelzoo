# TEXT=examples/translation/wmt14_en_de
# fairseq-preprocess \
#     --source-lang en --target-lang de \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0 \
#     --workers 20

# # Train the model
mkdir -p checkpoints/fconv_wmt_en_de
# export TORCH_SDAA_LOG_LEVEL=debug
# export SDAA_LAUNCH_BLOCKING=1
SDAA_VISIBLE_DEVICES=2,3 fairseq-train \
    data-bin/wmt14_en_de \
    --arch fconv_wmt_en_de \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000 \
    --save-dir checkpoints/fconv_wmt_en_de \
    --log-format json \
    --log-interval 2 \
    --log-file checkpoints/fconv_wmt_en_de/log.log \
    --amp \
    # | tee sdaa.log

# Evaluate
# fairseq-generate data-bin/wmt14_en_de \
#     --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
#     --beam 5 --remove-bpe