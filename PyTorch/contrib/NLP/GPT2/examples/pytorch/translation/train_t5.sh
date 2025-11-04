export SDAA_VISIBLE_DEVICES=2
export HF_ENDPOINT=https://hf-mirror.com
python examples/pytorch/translation/run_translation.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name wmt14/wmt14-en-de-pre-processed.py \
    --output_dir tmp/tst-translation \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --trust_remote_code True \
