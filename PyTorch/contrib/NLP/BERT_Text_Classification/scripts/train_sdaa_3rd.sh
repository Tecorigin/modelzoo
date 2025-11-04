cd ..
pip install requirements.txt
pip install -e .

export TASK_NAME=mrpc
export SDAA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/datasets/huggingface
timeout 2h python -m torch.distributed.launch --master_port=$((RANDOM+10000)) --nproc_per_node=4 examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20 \
  --output_dir tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --logging_steps 20 \
  --fp16 \
  | tee scripts/train_sdaa_3rd.log