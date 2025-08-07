#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"
# 检查必要的模型文件
model_path="/data/teco-data/QAnything/bge-m3"
required_files=("config.json" "tokenizer.json" "tokenizer_config.json" "special_tokens_map.json")
echo "检查模型文件..."
for file in "${required_files[@]}"; do    
    if [ ! -f "$model_path/$file" ]; then
            echo "错误：缺少必要的模型文件: $file"
            echo "请从 https://huggingface.co/BAAI/bge-m3 下载所需文件"        
    exit 1    
    fi
done
# 检查 PyTorch 版本和对应的模型文件
python_version=$(python -c "import torch; print(torch.__version__)")
if python -c "from packaging import version; import torch; exit(0 if version.parse(torch.__version__) < version.parse('2.6.0') else 1)"; then    
    if [ ! -f "$model_path/model.safetensors" ]; then
            echo "错误：PyTorch 版本 < 2.6.0，需要使用 model.safetensors 文件"
            echo "请从 https://huggingface.co/BAAI/bge-m3/tree/29fbe26ba08e5b2a5f06bac3632195bc85d3d690 下载 model.safetensors"        
            exit 1    
    fi
else    
if [ ! -f "$model_path/pytorch_model.bin" ]; then
        echo "错误：PyTorch 版本 >= 2.6.0，需要使用 pytorch_model.bin 文件"
        echo "请从 https://huggingface.co/BAAI/bge-m3 下载 pytorch_model.bin"        
        exit 1    
    fi
fi
# 安装依赖
echo "正在安装Python依赖..."
cd $script_path/
pip install -r requirements.txt

# 数据集路径设置
train_data_path="/data/teco-data/squad/qanything_train.jsonl"
# 模型路径设置
model_path="/data/teco-data/QAnything/bge-m3"

# 训练参数配置
log_file="$script_path/sdaa_train.log"
output_dir="$script_path/outputs/emb_m3_sdaa"

# 启动训练
echo "开始训练..."
cd $script_path/

# 环境变量设置（根据实际需求调整）
export TORCH_SDAA_AUTOLOAD=cuda_migrate
export TORCH_SDAA_RUNTIME_AUTOFALLBACK=1

python train.py \
    --train_file $train_data_path \
    --model_name_or_path $model_path \
    --output_dir $output_dir \
    --batch_size 6 \
    --max_steps 100 \
    --lr 1e-5 \
    --log_file "sdaa.log" \
    2>&1 | tee $log_file

# 生成loss曲线图（如果有对应的绘图脚本）
echo "生成训练结果图表..."
python loss.py 

echo "训练完成！结果保存在:"
echo " - 训练日志: $log_file"
echo " - 模型输出: $output_dir"
echo " - Loss曲线: $output_dir/loss.jpg"