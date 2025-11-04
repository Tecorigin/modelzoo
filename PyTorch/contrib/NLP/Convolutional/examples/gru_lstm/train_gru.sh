mkdir -p checkpoints/gru_iwslt_de_en

export SDAA_VISIBLE_DEVICES=0,1

fairseq-train data-bin/iwslt14.tokenized.de-en \
    --arch gru \
    --max-tokens 12000 \
    --save-dir checkpoints/gru_iwslt_de_en \
    --log-format json \
    --log-interval 20 \
    --log-file checkpoints/gru_iwslt_de_en/log.log \
    --max-epoch 20 \
    --criterion adaptive_loss \
    --adaptive-softmax-cutoff 1000 \
    --ddp-backend legacy_ddp \
    --optimizer adam \
    --lr 1e-5
    # --amp \