# CosyVoice 在 SDAA 上部署与推理 README

> 适用场景：你有一块基于 **SDAA** 的新显卡（TecoAI 系列），需要在服务器上部署并离线推理 **CosyVoice (FunAudioLLM)**。本文记录从零搭建环境、下载模型、修补依赖、完成推理输出 `cosy_out.wav` 的完整流程，以及常见报错的快速修复方法。

---

## 目录

* [0. 目录结构建议](#0-目录结构建议)
* [1. 环境准备](#1-环境准备)

  * [1.1 新建 Conda 环境](#11-新建-conda-环境)
  * [1.2 安装 Torch-SDAA 生态](#12-安装-torch-sdaa-生态)
  * [1.3 常用科学与音频依赖](#13-常用科学与音频依赖)
  * [1.4 CosyVoice 依赖（transformers 等）](#14-cosyvoice-依赖transformers-等)
* [2. 拉取源码 & 安装](#2-拉取源码--安装)
* [3. 下载模型权重](#3-下载模型权重)
* [4. 关键环境变量](#4-关键环境变量)
* [5. 推理脚本 `infer_sdaa.py`](#5-推理脚本-infer_sdaapy)
* [6. 运行示例](#6-运行示例)
* [7. 常见问题 & 速修表](#7-常见问题--速修表)
* [8. 性能优化建议](#8-性能优化建议)
* [9. 持久化与备份](#9-持久化与备份)
* [10. 扩展：零样本/VC/批量合成](#10-扩展零样本vc批量合成)

---

## 0. 目录结构建议

```
/data/bigc-data/zh/CosyVoice/
 ├─ models/                 # HF 下载的所有权重与配置
 │   ├─ cosyvoice.yaml
 │   ├─ llm.pt
 │   ├─ flow.pt
 │   ├─ hift.pt
 │   ├─ spk2info.pt
 │   ├─ campplus.onnx
 │   ├─ speech_tokenizer_v1.onnx
 │   └─ ...
 ├─ cosyvoice/              # 源码（git clone 或解压）
 ├─ infer_sdaa.py           # 最终推理脚本
 └─ cosy_out.wav            # 推理输出音频
```

---

## 1. 环境准备

### 1.1 新建 Conda 环境

```bash
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
```

### 1.2 安装 Torch-SDAA 生态

> **版本必须与驱动/Runtime 对齐**（示例：Torch 2.4.0a0、Torch-SDAA 2.1.0）。请替换成你们实际的 whl 源/本地包路径。

```bash
pip uninstall -y torch torch_sdaa tecodnn tecoblas sdaart sdpti tecodnn_ext || true
pip install -f <YOUR_WHL_INDEX_OR_DIR> \
  torch==2.4.0a0+git4451b0e \
  torch_sdaa==2.1.0 \
  tecodnn==2.1.0 tecoblas==2.1.0 sdaart==2.1.0 sdpti==1.4.0b0 tecodnn_ext==1.20.0a0 \
  --no-cache-dir
```

验证：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('has sdaa backend:', hasattr(torch.backends, 'sdaa'))
PY
```

### 1.3 常用科学与音频依赖

```bash
pip install numpy==1.26.4 scipy==1.11.4 soundfile librosa==0.9.2 torchaudio==2.4.0 --no-cache-dir
pip install onnxruntime==1.17.1 tqdm pyyaml hyperpyyaml==1.2.0 ruamel.yaml==0.17.32 --no-cache-dir
```

### 1.4 CosyVoice 依赖（transformers 等）

```bash
pip install --no-cache-dir \
  transformers==4.41.0 tokenizers==0.19.1 safetensors==0.4.3 \
  accelerate==0.30.1 sentencepiece==0.1.99 \
  huggingface_hub==0.29.1 diffusers==0.29.2

# 前端文本处理（WeTextProcessing 替代 ttsfrd/tn）
pip install WeTextProcessing==0.2.0 ttsfrd==0.4.3  # ttsfrd 可选

# 若日志报缺：
pip install pyarrow==15.0.2 datasets==2.19.0 pyworld==0.3.4 numba==0.58.1 llvmlite==0.41.1
```

---

## 2. 拉取源码 & 安装

```bash
cd /data/bigc-data/zh/CosyVoice
# 任选其一
# git clone https://github.com/FunAudioLLM/CosyVoice.git cosyvoice
# 或下载 zip 解压到 cosyvoice/

# 如果项目没有标准 setup.py，可以不安装，直接将该目录加入 PYTHONPATH。
# 有则：
# pip install -e cosyvoice
```

---

## 3. 下载模型权重

使用 HF CLI（或手动下载）放入 `models/`：

```bash
mkdir -p /data/bigc-data/zh/CosyVoice/models
cd /data/bigc-data/zh/CosyVoice/models
huggingface-cli download FunAudioLLM/CosyVoice-300M-SFT \
  --include "*.pt" "*.yaml" "*.json" "*.onnx" --local-dir .
```

确认以下关键文件存在：`cosyvoice.yaml`、`llm.pt`、`flow.pt`、`hift.pt`、`spk2info.pt`、`campplus.onnx`、`speech_tokenizer_v1.onnx`。

---

## 4. 关键环境变量

```bash
export SDAA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/tecoai/lib64:$LD_LIBRARY_PATH  # 根据你的 teco 库路径修改
```

---

## 5. 推理脚本 `infer_sdaa.py`

> 已替你解决：tn/pynini 英文 FST 导致的报错、流式生成器返回、音频字段不一致、SDAA autocast 等问题。

将下方脚本保存为：`/data/bigc-data/zh/CosyVoice/infer_sdaa.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice SDAA 推理脚本（最终稳定版）
"""

import os
import sys
import types
import argparse
import numpy as np
import torch
import soundfile as sf

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/data/bigc-data/zh/CosyVoice/models")
    p.add_argument("--text", default="欢迎使用CosyVoice。")
    p.add_argument("--spk", default="0")           # 索引或名字
    p.add_argument("--out", default="cosy_out.wav")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--device", default="auto", choices=["auto", "sdaa", "cuda", "cpu"])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ---------- Device ----------
def pick_device(pref):
    if pref == "sdaa" and hasattr(torch.backends, "sdaa"):
        return "sdaa"
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "cpu":
        return "cpu"
    # auto
    if hasattr(torch.backends, "sdaa"):
        return "sdaa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def safe_to_eval(m, device):
    try:
        if hasattr(m, "to"):   m.to(device)
        if hasattr(m, "eval"): m.eval()
    except Exception as e:
        print(f"⚠️ skip to/eval: {e}")

# ---------- Audio utils ----------
AUDIO_KEYS = ["wav", "wav_chunk", "audio", "speech", "samples", "waveform", "audio_16k", "tts_speech"]

def to_1d_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 0 or arr.size == 0:
        return None
    return arr.reshape(-1)

def extract_from_item(item, debug=False, idx=None):
    if isinstance(item, dict):
        if debug:
            print(f"[DEBUG] chunk#{idx} dict keys={list(item.keys())}")
        for k in AUDIO_KEYS:
            if k in item:
                v = item[k]
                if isinstance(v, dict):
                    for kk in AUDIO_KEYS:
                        if kk in v:
                            return to_1d_np(v[kk]), item.get("sr", v.get("sr", 22050))
                    return to_1d_np(v), item.get("sr", 22050)
                return to_1d_np(v), item.get("sr", 22050)
        return None, None
    if isinstance(item, (torch.Tensor, np.ndarray)):
        if debug:
            print(f"[DEBUG] chunk#{idx} tensor/ndarray shape={item.shape}")
        return to_1d_np(item), None
    if debug:
        print(f"[DEBUG] chunk#{idx} type={type(item)} ignored")
    return None, None

def collect_from_gen(gen, debug=False):
    bufs = []
    sr_final = 22050
    for i, it in enumerate(gen):
        arr, sr = extract_from_item(it, debug=debug, idx=i)
        if arr is not None:
            bufs.append(arr)
            if sr is not None:
                sr_final = sr
    if not bufs:
        raise ValueError("No valid audio chunk found in generator.")
    return np.concatenate(bufs, axis=0), sr_final

# ---------- Patch tn ----------
def patch_tn():
    fake_tn = types.ModuleType("tn")
    fake_tn.chinese = types.ModuleType("tn.chinese")
    fake_tn.english = types.ModuleType("tn.english")
    class _DummyNorm:
        def __init__(self, *a, **k): ...
        def normalize(self, text, *a, **k): return text
    fake_tn.chinese.normalizer = types.SimpleNamespace(Normalizer=_DummyNorm)
    fake_tn.english.normalizer = types.SimpleNamespace(Normalizer=_DummyNorm)
    sys.modules["tn"] = fake_tn
    sys.modules["tn.chinese"] = fake_tn.chinese
    sys.modules["tn.chinese.normalizer"] = fake_tn.chinese.normalizer
    sys.modules["tn.english"] = fake_tn.english
    sys.modules["tn.english.normalizer"] = fake_tn.english.normalizer

# ---------- Main ----------
def main():
    args = parse_args()

    os.environ.setdefault("SDAA_VISIBLE_DEVICES", "0")
    device = pick_device(args.device)
    print("Use device:", device)

    patch_tn()  # 必须在 import CosyVoice 前
    from cosyvoice.cli.cosyvoice import CosyVoice

    model = CosyVoice(model_dir=args.model_dir)

    for name in ["llm", "flow", "vocoder", "generator", "speech_tokenizer", "text_encoder"]:
        if hasattr(model, name):
            safe_to_eval(getattr(model, name), device)

    spk_list = model.list_available_spks()
    print("Available speakers:", spk_list)

    if args.spk.isdigit():
        idx = int(args.spk)
        if not (0 <= idx < len(spk_list)):
            raise ValueError(f"索引 {idx} 超界，0~{len(spk_list)-1}")
        spk_key = spk_list[idx]
    else:
        spk_key = args.spk
        if spk_key not in spk_list:
            raise ValueError(f"说话人 '{spk_key}' 不在列表：{spk_list}")

    import types as _types
    if isinstance(model.inference_sft, _types.MethodType):
        orig_fn = model.inference_sft.__func__
    else:
        orig_fn = model.inference_sft

    def inference_sft_nostream(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        res = orig_fn(self, tts_text, spk_id, stream=True, speed=speed, text_frontend=text_frontend)
        if isinstance(res, (_types.GeneratorType, list, tuple)):
            wav_np, sr = collect_from_gen(res, debug=args.debug)
            return {"wav": torch.from_numpy(wav_np), "sr": sr}
        if isinstance(res, dict):
            return res
        return {"wav": res, "sr": 22050}

    model.inference_sft = _types.MethodType(inference_sft_nostream, model)

    text = args.text
    use_fp16 = args.fp16 or (device != "cpu")
    dtype = torch.float16 if use_fp16 else torch.float32

    with torch.autocast(device_type=device, enabled=(device != "cpu"), dtype=dtype), torch.no_grad():
        out = model.inference_sft(
            tts_text=text,
            spk_id=spk_key,
            stream=False,
            speed=args.speed,
            text_frontend=True
        )

    if isinstance(out, dict):
        sr = out.get("sr", 22050)
        wav = None
        for k in AUDIO_KEYS:
            if k in out:
                wav = out[k]
                break
        if wav is None:
            raise RuntimeError(f"输出 dict 无音频字段：{list(out.keys())}")
    else:
        wav = out
        sr = 22050

    wav_np = to_1d_np(wav)
    if wav_np is None:
        raise RuntimeError("输出中没有有效音频数据，请加 --debug 检查。")

    sf.write(args.out, wav_np, sr)
    print(f"✅ Done -> {args.out}  (speaker={spk_key}, len={len(wav_np)/sr:.2f}s, sr={sr})")

if __name__ == "__main__":
    main()
```

---

## 6. 运行示例

```bash
conda activate cosyvoice
export SDAA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/tecoai/lib64:$LD_LIBRARY_PATH

python infer_sdaa.py \
  --model-dir /data/bigc-data/zh/CosyVoice/models \
  --text "欢迎使用CosyVoice。" \
  --spk 中文女 \
  --out cosy_out.wav \
  --device sdaa
```

看到输出：

```
✅ Done -> cosy_out.wav  (speaker=中文女, len=1.xx s, sr=22050)
```

即表示成功。

---

## 7. 常见问题 & 速修表

| 报错 / 现象                                    | 解决方式                                                                          |
| ------------------------------------------ | ----------------------------------------------------------------------------- |
| `ModuleNotFoundError: hyperpyyaml`         | `pip install hyperpyyaml ruamel.yaml`                                         |
| `No module named 'transformers'`           | `pip install transformers tokenizers safetensors`                             |
| `_ZNK5torch8autograd4Node4name...`         | torch / torch\_sdaa 版本不匹配；删 `import torch_sdaa`；统一重装 whl                      |
| `tn.* FstIOError: Read failed`             | 使用 monkey patch `patch_tn()` 或手动注释英文 normalizer                               |
| `Generator returned no valid audio chunks` | 用脚本里 collect\_from\_gen；加 `--debug` 看返回字段，补 AUDIO\_KEYS                       |
| `ValueError: zero-dimensional arrays`      | 同上，过滤空块；脚本已处理                                                                 |
| 速度非常慢（几分钟一句）                               | 关闭流式；改为 `torch.autocast(device_type="sdaa")`；检查 vocoder/llm 是否上卡；调大 `--speed` |
| `pyarrow/pyworld` 缺失                       | `pip install pyarrow pyworld numba llvmlite` 或猴子补丁 dataset/processor          |
| FutureWarning: `weights_only=False`        | 可忽略或改源码 `torch.load(..., weights_only=True)`                                  |

---

## 8. 性能优化建议

* **autocast**：全部统一 `torch.autocast(device_type="sdaa", dtype=torch.float16)`。
* **关闭 tqdm/日志**：减少 Python 开销。
* **速度参数**：`--speed 1.2~1.4` 视音质容忍度调节。
* **模型上卡检测**：打印 `next(model.flow.parameters()).device` 等确认不在 CPU。

---

## 9. 持久化与备份

```bash
# 导出环境
conda env export --no-builds > cosyvoice_env.yaml
# 下次直接
conda env create -f cosyvoice_env.yaml
```

也可以把 `infer_sdaa.py`、`models` 整体打包。

---

## 10. 扩展：零样本/VC/批量合成

* **零样本克隆**：`model.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, ...)`
* **跨语言**：`model.inference_cross_lingual(...)`
* **语音转换 VC**：`model.inference_vc(source_speech_16k, prompt_speech_16k, ...)`

如需对应脚本/接口或 WebUI/Gradio Demo，可再告诉我。

---

**到此为止，你已经可以在 SDAA 上稳定地跑通 CosyVoice 了。祝使用愉快！** 🎧
