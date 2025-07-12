# YOLOv6(nsmlx) 推理使用说明

> 本文档基于 **Ultralytics 8.3.163** 生态，锁定 `numpy 1.26.x` 与 `opencv-python<4.10`，并演示如何在 **GPU (CUDA 12.1)** 环境下快速完成权重下载、ONNX 导出、推理及精度验证。


## 1. 模型概述

YOLOv6 由美团视觉团队提出，采用轻量级 **EfficientRep** 骨干与自研 **Rep-PAN** 颈部，兼顾速度与精度，广泛用于工业检测场景。本示例选用 **YOLOv6‑S**（COCO 预训练），输入 640×640 时 **mAP50≈43.1%**，推理速度 ≈ 150 FPS（Tesla A100，batch = 1，FP16）。

---

## 2. 快速开始

准备在 **`~/yolov6`** 目录运行。

```bash
# 克隆仓库并进入
mkdir ~/yolov6 && cd ~/yolov6
```

### 2.1 基础环境安装

```bash
# (可跳过) 创建并激活 Conda 环境
conda create --name yolo --clone torch_env_py310
conda activate yolov

# 安装核心依赖（锁定版本，避免 NumPy、OpenCV 冲突）
pip install "numpy>=1.23,<2.0" "opencv-python<4.10" \
            torch==2.4.0+cu121 -f https://download.pytorch.org/whl/cu121
# 安装 Ultralytics 主包（禁止自动升级依赖）
pip install ultralytics==8.3.163 --no-deps
```

### 2.2 安装第三方依赖

```bash
# (可选) 使用 SDAA 推理时需额外装定制包
pip install torch-sdaa==2.1.0 --force-reinstall

# (可选) TensorRT 推理
pip install tensorrt==10.0.1 onnxsim==0.4.35
```

> **注意**：若手动修改 `--model` 路径，请确保指向有效 `.pt / .onnx` 文件。

### 2.3 获取 ONNX 文件

1. **权重下载**（自动）：

   ```bash
   # Ultralytics 会在首次调用时把 yolov6s.pt 下载到 ~/.cache/ultralytics
   yolo predict model=yolov6s.pt source=bus.jpg --dryrun
   ```
2. **导出脚本**（`export_onnx.py`）：

   ```python
   from ultralytics import YOLO
   YOLO('yolov6s.pt').export(format='onnx', imgsz=640, opset=13, simplify=True)
   ```

   运行后将在 `yolov6s.onnx` 同级目录生成：

   * `yolov6s.onnx`  (检测)
   * `yolov6s.yaml`  (模型结构元数据)

   **主要参数**

   | 参数         | 说明                | 默认   |
   | ---------- | ----------------- | ---- |
   | `imgsz`    | 输入尺寸              | 640  |
   | `opset`    | ONNX opset 版本     | 13   |
   | `simplify` | 是否调用 onnxsim 做图优化 | True |

### 2.4 获取数据集

| 数据集           | 用途   | 下载                                      | 处理脚本                 |
| ------------- | ---- | --------------------------------------- | -------------------- |
| COCO 2017 val | 精度验证 | [官网](https://cocodataset.org/#download) | `tools/coco2yolo.py` |

示例快速下载（仅验证集）：

```bash
mkdir -p data && cd data
wget -c https://images.cocodataset.org/zips/val2017.zip
wget -c https://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip && unzip annotations_trainval2017.zip
cd ..
```

### 2.5 启动推理

#### 2.5.1 单张图片 (PyTorch)

```bash
python - <<'PY'
from ultralytics import YOLO
m = YOLO('yolov6s.pt').to('cuda')
res = m('bus.jpg', imgsz=640, conf=0.25)
res[0].save('bus_vis.png')
PY
```

#### 2.5.2 文件夹批量 (ONNX + onnxruntime)

```bash
pip install onnxruntime-gpu==1.18.0

python infer_onnx.py \
  --model yolov6s.onnx \
  --source val2017 --output runs/v6_onnx \
  --imgsz 640 --conf 0.25 --device 0
```

`infer_onnx.py` 提供：

* `--device`        GPU id；填 `cpu` 即在 CPU 上跑
* `--save-txt`      同步导出 YOLO txt 结果
* `--half`          FP16 加速 (需 GPU)

### 2.6 精度验证

```bash
# PyTorch 模式评估 COCO mAP
python - <<'PY'
from ultralytics import YOLO
m = YOLO('yolov6s.pt')
metrics = m.val(data='coco.yaml', split='val', batch=32, device=0, imgsz=640)
print(metrics)
PY
```

> 期望输出：`mAP50 ≈ 0.431`，与官方公布一致即为正常。

---

## 参考

* [YOLOv6 论文 & 代码](https://github.com/meituan/YOLOv6)
* [Ultralytics Hub](https://hub.ultralytics.com/) – 在线权重管理
* [ONNX Runtime GPU](https://onnxruntime.ai/) – 推理加速

如有问题，请在 Issues 提问或提交 PR 改进文档。
