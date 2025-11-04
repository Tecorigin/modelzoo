export LD_LIBRARY_PATH=/softwares/opt/tecoai/lib64:$LD_LIBRARY_PATH
export SDAA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
python -m torch.distributed.launch --master_port=$((RANDOM+10000)) --nproc_per_node=4 examples/pytorch/translation/run_translation.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name wmt14/wmt14-en-de-pre-processed.py \
    --output_dir tmp/test-translation \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --trust_remote_code True \
    --logging_steps 200 \
    --fp16 \
