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
# OF SUCH DAM
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an embedding model (e.g. BAAI/bge-m3) on SDAA for a few steps and log loss.

- 单卡即可运行（如需多卡可自行加 DDP）
- 默认对 CLS 向量做一个简单的对比损失（InfoNCE）
- 每步将 loss 追加到 sdaa_loss.log
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

# ---------------------- Dataset ---------------------- #
class QADataset(Dataset):
    """读取 jsonl，每行包含 question/context，拼接后做无监督训练示例。"""
    def __init__(self, jsonl_path, tokenizer, max_len=512):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.samples = [json.loads(l) for l in f]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        q = self.samples[i].get("question", "")
        ctx = self.samples[i].get("context", "")
        return q + " " + ctx

def collate_fn(batch, tok, max_len):
    return tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

# ---------------------- Loss ---------------------- #
def contrastive_loss(emb, temperature=0.05):
    """
    简单 InfoNCE：用自身 batch 作为正/负样本。
    仅用于演示，真实场景请使用更合理的正负样本构建。
    """
    emb = torch.nn.functional.normalize(emb, dim=-1)
    sim = emb @ emb.t() / temperature          # [B, B]
    labels = torch.arange(sim.size(0), device=sim.device)
    return torch.nn.functional.cross_entropy(sim, labels)

# ---------------------- Main ---------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="本地模型目录，比如 /data/bigc-data/zh/QAnything/bge-m3")
    parser.add_argument("--output_dir", type=str, default="./outputs/emb_m3_sdaa")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--log_file", type=str, default="sdaa_loss.log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="关闭 bfloat16 自动混合精度")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("sdaa")

    # 关闭联网，只用本地文件
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, local_files_only=True
    ).to(device)
    model.train()

    dataset = QADataset(args.train_file, tokenizer, args.max_len)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    log_path = os.path.join(args.output_dir, args.log_file)
    with open(log_path, "w", buffering=1) as fp:
        fp.write("step,loss\n")

        step = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="sdaa", dtype=torch.bfloat16, enabled=not args.no_amp):
                out = model(**batch, output_hidden_states=False)
                cls_emb = out.last_hidden_state[:, 0]  # [B, D]
                loss = contrastive_loss(cls_emb)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            fp.write(f"{step},{loss.item():.6f}\n")

            if step >= args.max_steps:
                break

    # 保存权重（可选）
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Loss log -> {log_path}")

if __name__ == "__main__":
    main()
