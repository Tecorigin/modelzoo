cd ..
pip install -r requirements.txt
pip install -e .
export ROBERTA_PATH=/data/datasets/roberta/pretrained/model.pt
export SDAA_VISIBLE_DEVICES=0,1,2,3  

timeout 5h fairseq-hydra-train -m --config-dir examples/roberta/config/finetuning --config-name rte \
task.data=/data/datasets/RTE-bin checkpoint.restore_file=$ROBERTA_PATH | tee scripts/train_sdaa_3rd.log


