export SDAA_VISIBLE_DEVICES=0,1

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  --amp \
  --save-dir checkpoints/hubert \
  --log-format json \
  --log-interval 2 \
  --log-file checkpoints/hubert/log.log \
  task.data=/path/to/data task.label_dir=/path/to/labels task.labels='["km"]' model.label_rate=100