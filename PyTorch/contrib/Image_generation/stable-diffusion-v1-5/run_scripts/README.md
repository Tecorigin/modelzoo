# Stable Diffusion Fine-tuning Training

## 参数说明

| 参数名         | 说明                                | 示例              |
| -------------- | ----------------------------------- | ----------------- |
| `--model_name` | 模型名称，暂未使用，可扩展          | `sd_unet`         |
| `--batch_size` | 训练批大小，影响显存和速度          | `1`               |
| `--max_iter`   | 最大训练迭代次数                    | `100`             |
| `--device`     | 训练设备，如 `sdaa:0` 或 `cuda:0`  | `sdaa:0`          |
| `--data_size`  | 训练用数据集大小（样本数量）        | `2000`            |
| `--log_path`   | 训练日志文件保存路径                | `sdaa.log`        |
| `--save_path`  | 训练完成后模型保存路径              | `unet_finetuned.pth` |
| `--accum_steps`| 梯度累积步数                       | `2`               |
| `--coco_img_root` | COCO 图片根目录，必须指定           | `/path/to/train2017` |
| `--coco_ann_path` | COCO 注释文件路径，必须指定         | `/path/to/captions_train2017.json` |

---

## 运行示例

```bash
python run_scripts/run_demo.py \
  --batch_size 1 \
  --max_iter 100 \
  --device sdaa:0 \
  --data_size 2000 \
  --coco_img_root /path/to/train2017 \
  --coco_ann_path /path/to/captions_train2017.json \
  --log_path /path/to/sdaa.log \
  --save_path /path/to/unet_finetuned.pth \
  --accum_steps 2
