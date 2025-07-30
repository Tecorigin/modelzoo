# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMA
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice SDAA 推理脚本（最终稳定版）
"""

import os
import sys
import types
import argparse
from pathlib import Path

import numpy as np
import torch
import soundfile as sf


# --------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/data/bigc-data/zh/CosyVoice/models",
                   help="包含 cosyvoice.yaml / llm.pt / flow.pt / hift.pt / campplus.onnx 等文件的目录")
    p.add_argument("--text", default="欢迎使用CosyVoice。", help="要合成的文本")
    p.add_argument("--spk", default="0", help="说话人（索引 0/1/... 或名字 '中文女' 等）")
    p.add_argument("--out", default="cosy_out.wav", help="输出 wav 文件路径")
    p.add_argument("--speed", type=float, default=1.0, help="语速")
    p.add_argument("--device", default="auto", choices=["auto", "sdaa", "cuda", "cpu"], help="推理设备")
    p.add_argument("--fp16", action="store_true", help="强制 FP16（默认：GPU/SDAA 自动 FP16）")
    p.add_argument("--debug", action="store_true", help="打印 chunk 调试信息")
    return p.parse_args()


# ------------------ Device Utils ------------------
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
        if hasattr(m, "to"):
            m.to(device)
        if hasattr(m, "eval"):
            m.eval()
    except Exception as e:
        print(f"⚠️ skip to/eval: {e}")


# ------------------ Audio Utils ------------------
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
    # dict
    if isinstance(item, dict):
        if debug:
            print(f"[DEBUG] chunk#{idx} dict keys={list(item.keys())}")
        for k in AUDIO_KEYS:
            if k in item:
                v = item[k]
                # 嵌套 dict 再搜
                if isinstance(v, dict):
                    for kk in AUDIO_KEYS:
                        if kk in v:
                            return to_1d_np(v[kk]), item.get("sr", v.get("sr", 22050))
                    return to_1d_np(v), item.get("sr", 22050)
                return to_1d_np(v), item.get("sr", 22050)
        return None, None
    # tensor/ndarray
    if isinstance(item, (torch.Tensor, np.ndarray)):
        if debug:
            print(f"[DEBUG] chunk#{idx} tensor/ndarray shape={item.shape}")
        return to_1d_np(item), None
    # 其他
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


# ------------------ TN Monkey Patch ------------------
def patch_tn():
    """伪造 tn 包，避免 import tn.english/tn.chinese 触发 FST 错误。"""
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


# ------------------ Main ------------------
def main():
    args = parse_args()

    os.environ.setdefault("SDAA_VISIBLE_DEVICES", "0")
    device = pick_device(args.device)
    print("Use device:", device)

    # 1) 先 patch tn，再 import CosyVoice
    patch_tn()
    from cosyvoice.cli.cosyvoice import CosyVoice

    # 2) 初始化模型
    model = CosyVoice(model_dir=args.model_dir)

    # 3) 子模块迁移
    for name in ["llm", "flow", "vocoder", "generator", "speech_tokenizer", "text_encoder"]:
        if hasattr(model, name):
            safe_to_eval(getattr(model, name), device)

    spk_list = model.list_available_spks()
    print("Available speakers:", spk_list)

    # 4) 说话人处理
    if args.spk.isdigit():
        idx = int(args.spk)
        if not (0 <= idx < len(spk_list)):
            raise ValueError(f"索引 {idx} 超界，0~{len(spk_list)-1}")
        spk_key = spk_list[idx]
    else:
        spk_key = args.spk
        if spk_key not in spk_list:
            raise ValueError(f"说话人 '{spk_key}' 不在列表：{spk_list}")

    # 5) 猴子补丁 inference_sft -> 一次性返回 dict
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

    # 6) 推理
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

    # 7) 取 wav
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
