set -e  # 遇错退出

# === 配置参数 ===
MODEL_NAME="VGG19"
BATCH_SIZE=128
EPOCHS=250
LR=0.01
RESUME=""

# === 可选：使用 --resume 参数 ===
# RESUME="--resume"

echo "[INFO] 开始训练模型: $MODEL_NAME"
python run_scripts/run_demo.py \
  --model_name $MODEL_NAME \
  --batchsize $BATCH_SIZE \
  --epoch $EPOCHS \
  --lr $LR \
  $RESUME

# === 训练完成后进行loss对比图绘制 ===
echo "[INFO] 正在生成 loss 对比图..."
python loss.py

echo "[INFO] 全部任务完成 ✅ 输出图像: loss_comparison.png"