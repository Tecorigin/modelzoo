export SDAA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com

python -m torch.distributed.launch --master_port=$((RANDOM+10000)) --nproc_per_node=4 examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path openai-community/gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --do_train \
    --do_eval \
    --output_dir tmp/test-clm \
    --overwrite_output_dir \
    --logging_steps 10 \
    --fp16 \