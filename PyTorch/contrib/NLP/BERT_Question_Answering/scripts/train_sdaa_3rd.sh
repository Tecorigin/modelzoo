cd ..
pip insstall -r requirements.txt
pip install -e .

export SDAA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/datasets/huggingface
timeout 2h python -m torch.distributed.launch --master_port=$((RANDOM+10000)) --nproc_per_node=4 examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 6 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir tmp/qa_squad/ \
  --overwrite_output_dir \
  --logging_steps 100 \
  --fp16 \
  | tee scripts/train_sdaa_3rd.log