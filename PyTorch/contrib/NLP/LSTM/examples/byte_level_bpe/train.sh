VOCAB=bbpe2048
export SDAA_VISIBLE_DEVICES=0,1
mkdir -p checkpoints/byte_level_bpe

fairseq-train "data/bin_${VOCAB}" --task translation --user-dir examples/byte_level_bpe/gru_transformer \
    --arch gru_transformer --encoder-layers 2 --decoder-layers 2 --dropout 0.3 --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --log-format json --log-interval 20 --save-dir checkpoints/byte_level_bpe \
    --log-file checkpoints/byte_level_bpe/log.log \
    --batch-size 100 --max-update 100000 --update-freq 2