cd ..
mkdir -p checkpoints/lstm_iwslt_de_en
pip install -r requirements.txt
pip install -e .

export SDAA_VISIBLE_DEVICES=0,1,2,3

timeout 2h fairseq-train /data/datasets/iwslt14.tokenized.de-en \
    --arch lstm2 \
    --max-tokens 2000 \
    --save-dir checkpoints/lstm_iwslt_de_en \
    --log-format json \
    --log-interval 10 \
    --log-file scripts/train_sdaa_3rd.log \
    --max-epoch 20 \
    --criterion adaptive_loss \
    --adaptive-softmax-cutoff 1000 \
    --ddp-backend legacy_ddp \
    --optimizer adam \
    --lr 1e-5 \
    --amp \