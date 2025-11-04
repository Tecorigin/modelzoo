export SDAA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com



python examples/pytorch/text-generation/run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=openai-community/gpt2 \
    --fp16 \