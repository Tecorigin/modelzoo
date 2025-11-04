mkdir -p checkpoints/lstm_iwslt_de_en

export SDAA_VISIBLE_DEVICES=0,1

fairseq-train data-bin/iwslt14.tokenized.de-en \
    --arch lstm2 \
    --max-tokens 2000 \
    --save-dir checkpoints/lstm_iwslt_de_en \
    --log-format json \
    --log-interval 10 \
    --log-file checkpoints/lstm_iwslt_de_en/log.log \
    --max-epoch 20 \
    --criterion adaptive_loss \
    --adaptive-softmax-cutoff 1000 \
    --ddp-backend legacy_ddp \
    --optimizer adam \
    --lr 1e-5 \
    --amp \