cd ..
pip install -r requirements.txt
pip install -e .

export SDAA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/datasets/huggingface
timeout 2h python -m torch.distributed.launch --master_port=$((RANDOM+10000)) --nproc_per_node=4 examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir tmp/test-summarization \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --logging_steps 100 \
    --fp16 \
    | tee scripts/train_sdaa_3rd.log
