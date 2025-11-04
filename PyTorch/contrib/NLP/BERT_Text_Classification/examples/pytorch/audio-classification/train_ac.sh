export SDAA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com

# python examples/pytorch/audio-classification/run_audio_classification.py \
#     --model_name_or_path facebook/wav2vec2-base \
#     --dataset_name superb/superb.py \
#     --dataset_config_name ks \
#     --output_dir tmp/wav2vec2-base-ft-keyword-spotting \
#     --overwrite_output_dir \
#     --remove_unused_columns False \
#     --do_train \
#     --do_eval \
#     --fp16 \
#     --learning_rate 3e-5 \
#     --max_length_seconds 1 \
#     --attention_mask False \
#     --warmup_ratio 0.1 \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 32 \
#     --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 32 \
#     --dataloader_num_workers 4 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --eval_strategy epoch \
#     --save_strategy epoch \
#     --load_best_model_at_end True \
#     --metric_for_best_model accuracy \
#     --save_total_limit 3 \
#     --seed 0 \
#     --trust_remote_code True \

python examples/pytorch/audio-classification/run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_name common_language \
    --audio_column_name audio \
    --label_column_name language \
    --output_dir tmp/wav2vec2-base-lang-id \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 \
    --learning_rate 3e-4 \
    --max_length_seconds 16 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 3 \
    --seed 0 \
    --trust_remote_code True \
