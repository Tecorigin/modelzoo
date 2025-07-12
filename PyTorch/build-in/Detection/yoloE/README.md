# YOLOE 推理使用说明

> 本文档基于 **Ultralytics 8.3.163** 生态，锁定 `numpy 1.26.x` 与 `opencv-python<4.10` 环境，演示如何对 **YOLOE** 模型进行 ONNX 导出、推理及精度验证。

---

# 算法名称

**YOLOE** — 多任务目标检测与实例分割模型，支持 Prompt-Free 和 Prompt-Based 两种模式。

## 1. 模型概述

YOLOE 结合轻量级 **EfficientViT** 背骨、**BiFPN** 颈部和专用的 **DetectE**／**SegmentE** 头部，实现单网络同时完成检测和分割任务；同时支持文本或图像提示（Prompt-Based）与 4585 类内置词表（Prompt-Free），适用于通用场景下的高效视觉感知。

## 2. 快速开始

以下流程在 **`~/yoloe_demo`** 目录下执行。

```bash
mkdir -p ~/yoloe_demo && cd ~/yoloe_demo
```

### 2.1 基础环境安装

```bash
# (可选) 创建并激活 Conda 环境
conda create --name yolo --clone torch_env_py310
conda activate yoloe

# 安装核心依赖：
# 锁定 numpy、opencv 版本，确保与 torch-sdaa 兼容
pip install "numpy>=1.23,<2.0" "opencv-python<4.10" \
            torch==2.4.0+cu121 -f https://download.pytorch.org/whl/cu121

# 安装 Ultralytics 主包（禁止自动解析依赖）
pip install ultralytics==8.3.163 --no-deps
```

### 2.2 安装第三方依赖

```bash
# 如需 SDAA 推理
pip install torch-sdaa==2.1.0 --force-reinstall
# 如需 ONNX Runtime 加速
pip install onnxruntime-gpu==1.18.0 onnxsim==0.4.35
```

> **注意**：确保下面命令中的 `--model` 路径指向 `yoloe-11l-seg.pt` 或 `yoloe-11l-pf.pt` / `.onnx`。

### 2.3 获取 ONNX 文件

1. **下载预训练权重**（自动）：

   ```bash
   yolo predict model=yoloe-11l-seg.pt source=img.jpg --dryrun
   ```
2. **导出脚本** (`export_onnx.py`)：

   ```python
   from ultralytics import YOLOE
   YOLOE('yoloe-11l-seg.pt').export(
       format='onnx', imgsz=640, opset=13, simplify=True
   )
   ```

   运行后生成：

   * `yoloe-11l-seg.onnx`
   * `yoloe-11l-seg.yaml`

   **主要参数**

   | 参数         | 说明            | 默认   |
   | ---------- | ------------- | ---- |
   | `imgsz`    | 输入分辨率         | 640  |
   | `opset`    | ONNX opset 版本 | 13   |
   | `simplify` | 是否调用 onnxsim  | True |

### 2.4 获取数据集

| 数据集       | 用途      | 下载链接                                                                   | 处理脚本                 |
| --------- | ------- | ---------------------------------------------------------------------- | -------------------- |
| COCO 2017 | 检测 & 分割 | [https://cocodataset.org/#download](https://cocodataset.org/#download) | `tools/coco2yolo.py` |

示例下载：

```bash
mkdir -p data && cd data
wget https://images.cocodataset.org/zips/val2017.zip
wget https://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip && unzip annotations_trainval2017.zip
cd ..
```

### 2.5 启动推理

#### 2.5.1 单张图片（Prompt-Free，实例分割）

```bash
python - <<'PY'
from ultralytics import YOLOE
model = YOLOE('yoloe-11l-pf.pt').to('cuda')
res = model('image.jpg', imgsz=640, conf=0.25)
res[0].save('image_pf_seg.png')
PY
```

#### 2.5.2 批量 ONNX 推理（Prompt-Based，检测）

```bash
python infer_onnx.py \
  --model yoloe-11l-seg.onnx \
  --source data/val2017 \
  --output runs/yoloe_onnx \
  --imgsz 640 --conf 0.25 --device 0 --half
```

### 2.6 精度验证

#### 检测 mAP 验证

```bash
python - <<'PY'
from ultralytics import YOLOE
model = YOLOE('yoloe-11l.pt')
metrics = model.val(
    data='coco.yaml', split='val', imgsz=640, batch=16, device=0
)
print('Detection:', metrics)
PY
```

#### 分割 mAP 验证

```bash
python - <<'PY'
from ultralytics import YOLOE
data = 'coco-seg.yaml'  # 分割任务配置
model = YOLOE('yoloe-11l-seg.pt')
metrics = model.val(
    data=data, split='val', imgsz=640, batch=8, device=0
)
print('Segmentation:', metrics)
PY
```

> 期望检测 mAP50≈45%、分割 mAP50≈38%，如有偏差请检查环境和权重一致性。

---

## 参考

* [Ultralytics YOLOE 文档](https://github.com/ultralytics/ultralytics)
* [ONNX Runtime](https://onnxruntime.ai/)
* [COCO Dataset](https://cocodataset.org/)
