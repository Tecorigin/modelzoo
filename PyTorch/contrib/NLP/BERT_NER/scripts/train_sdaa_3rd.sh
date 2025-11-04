cd ..
pip install -r requirements.txt
pip install -e .

export SDAA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/datasets/huggingface
timeout 2h python -m torch.distributed.launch --nproc_per_node=4 examples/pytorch/token-classification/run_ner.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name conll2003/conll2003.py \
  --output_dir tmp/test-ner \
  --do_train \
  --do_eval \
  --trust_remote_code True \
  --overwrite_output_dir \
  --logging_steps 100 \
  --fp16 \
  | tee scripts/train_sdaa_3rd.log