cd ..
pip install -r requirements.txt
pip install -e .

TOTAL_NUM_UPDATES=2036  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=61      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=16        # Batch size.
BART_PATH=examples/bart/base/model.pt
mkdir -p checkpoints/bart
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
tar -xzvf bart.base.tar.gz -C examples/bart
python examples/bart/edit_ckpt.py
export SDAA_VISIBLE_DEVICES=0,1,2,3 
timeout 6h fairseq-train /data/datasets/RTE-bin/ \
    --restore-file $BART_PATH \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --add-prev-output-tokens \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --arch bart_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --save-dir checkpoints/bart \
    --log-format json \
    --log-interval 2 \
    --log-file scripts/train_sdaa_3rd.log \
    --amp

    # --log-file checkpoints/bart/log.log \