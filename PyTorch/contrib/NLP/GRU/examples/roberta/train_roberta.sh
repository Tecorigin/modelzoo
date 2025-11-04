export ROBERTA_PATH=examples/roberta/pretrained/model.pt
export SDAA_VISIBLE_DEVICES=0,1  

fairseq-hydra-train -m --config-dir examples/roberta/config/finetuning --config-name rte \
task.data=/root/cas/lzk/fairseq/RTE-bin checkpoint.restore_file=$ROBERTA_PATH


