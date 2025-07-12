# YOLOv12(nsmlx) 推理使用说明

> 本文档基于 **Ultralytics 8.3.163** 生态，锁定 `numpy 1.26.x` 与 `opencv-python<4.10` 环境，演示如何对 **YOLOv12** 模型进行 ONNX 导出、推理及精度验证。

---

# 算法名称

**YOLOv12** — Attention-Centric 目标检测模型，小型版本，兼具速度与精度。

## 1. 模型概述

YOLOv12 由 Ultralytics 推出，以自研 **EdgeFormer** 骨干与 **PAFPN** 颈部为核心，融合 CBAM 注意力模块和 C3 结构，适合实时检测。YOLOv12-N 在 COCO 数据集上可达 **mAP50≈44.5%**，单卡 GPU 推理速度约 **200 FPS**（A100，FP16，batch=1）。

## 2. 快速开始

以下流程在 **`~/yolov12`** 目录下执行。

```bash
mkdir -p ~/yolov12 && cd ~/yolov12
```

### 2.1 基础环境安装

```bash
# 可选：创建并激活 Conda 环境
conda create --name yolo --clone torch_env_py310
conda activate yolov12

# 安装核心依赖
pip install "numpy>=1.23,<2.0" "opencv-python<4.10" \
            torch==2.4.0+cu121 -f https://download.pytorch.org/whl/cu121
# 安装 Ultralytics（禁止自动解析依赖）
pip install ultralytics==8.3.163 --no-deps
```

### 2.2 安装第三方依赖

```bash
# 如需 SDAA 推理
pip install torch-sdaa==2.1.0 --force-reinstall
# 如需 ONNX Runtime 加速
pip install onnxruntime-gpu==1.18.0 onnxsim==0.4.35
```

> **注意**：请确保 `--model` 路径指向 `yolov12n.pt` 或 `.onnx` 文件。

### 2.3 获取 ONNX 文件

1. **下载预训练权重**（自动）：

   ```bash
   yolo predict model=yolov12n.pt source=bus.jpg --dryrun
   ```
2. **导出脚本**（`export_onnx.py`）：

   ```python
   from ultralytics import YOLO
   YOLO('yolov12n.pt').export(format='onnx', imgsz=640, opset=13, simplify=True)
   ```

   运行后生成：

   * `yolov12n.onnx`
   * `yolov12n.yaml` （结构配置）

   **主要参数**

   | 参数         | 说明            | 默认   |
   | ---------- | ------------- | ---- |
   | `imgsz`    | 输入分辨率         | 640  |
   | `opset`    | ONNX opset 版本 | 13   |
   | `simplify` | 是否简化图         | True |

### 2.4 获取数据集

| 数据集       | 用途   | 下载链接                                                                   | 处理脚本                 |
| --------- | ---- | ---------------------------------------------------------------------- | -------------------- |
| COCO 2017 | 精度验证 | [https://cocodataset.org/#download](https://cocodataset.org/#download) | `tools/coco2yolo.py` |

示例下载：

```bash
mkdir -p data && cd data
wget https://images.cocodataset.org/zips/val2017.zip
wget https://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip && unzip annotations_trainval2017.zip
cd ..
```

### 2.5 启动推理

#### 单张图片（PyTorch）

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolov12n.pt').to('cuda')
res = model('bus.jpg', imgsz=640, conf=0.25)
res[0].save('bus_v12.png')
PY
```

#### ONNX 推理

```bash
yolo predict model=yolo12.onnx source=bus.jpg imgsz=640 conf=0.25 save=True 
```

`infer_onnx.py` 支持：

* `--save-txt` 导出 txt 结果
* `--half` FP16 推理

### 2.6 精度验证

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolov12n.pt')
metrics = model.val(data='coco.yaml', split='val', imgsz=640, batch=16, device=0)
print(metrics)
PY
```

> 预期 `mAP50 ≈ 0.445`，若一致则推理与模型权重正常。

---

## 参考

* [YOLOv12 官方仓库](https://github.com/ultralytics/ultralytics)
* [ONNX Runtime](https://onnxruntime.ai/)
* [COCO Dataset](https://cocodataset.org/)
