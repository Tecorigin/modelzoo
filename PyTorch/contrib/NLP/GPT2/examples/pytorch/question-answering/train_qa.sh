export SDAA_VISIBLE_DEVICES=1,2
export HF_ENDPOINT=https://hf-mirror.com
python -m torch.distributed.launch --master_port=$((RANDOM+10000)) --nproc_per_node=2 examples/pytorch/question-answering/run_qa.py \
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
  --fp16