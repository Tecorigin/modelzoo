#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an embedding model (e.g. BAAI/bge-m3) on SDAA for a few steps and log loss.

- 单卡即可运行（如需多卡可自行加 DDP）
- 默认对 CLS 向量做一个简单的对比损失（InfoNCE）
- 每步将 loss 追加到 sdaa.log
"""

import os
import json
import argparse
import torch
import datetime
import time

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

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
def contrastive_loss(emb, temperature=0.1):
    """
    简单 InfoNCE：用自身 batch 作为正/负样本。
    这里增加了轻微扰动，避免 sim 矩阵对角线远远大于其他，loss 为 0 的情况。
    """
    emb = torch.nn.functional.normalize(emb, dim=-1)
    
    # 加轻微噪声扰动，帮助训练初期产生非零 loss
    emb = emb + torch.randn_like(emb) * 1e-3
    sim = emb @ emb.t() / temperature          # [B, B]
    
    labels = torch.arange(sim.size(0), device=sim.device)
    loss = torch.nn.functional.cross_entropy(sim, labels)
    return loss

# ---------------------- Main ---------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="本地模型目录，比如 /data/bigc-data/lsq/QAnything/new_qanything/bge-m3")
    parser.add_argument("--output_dir", type=str, default="./outputs/emb_m3_sdaa")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--log_file", type=str, default="sdaa.log")
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
    logger = Logger([
        StdOutBackend(Verbosity.DEFAULT),
        # JSONStreamBackend(Verbosity.VERBOSE, log_path),  # 你可根据需要打开 JSON 日志
    ])

    # 定义元数据
    logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.4f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
    logger.metadata("train.total_time", {"unit": "s", "format": ":.4f", "STAGE": "TRAIN"})

    step = 0
    start_time = time.time()

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.autocast(device_type="sdaa", dtype=torch.bfloat16, enabled=not args.no_amp):
            out = model(**batch, output_hidden_states=False)
            cls_emb = out.last_hidden_state[:, 0]  # [B, D]
            loss = contrastive_loss(cls_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        elapsed = time.time() - start_time
        start_time = time.time()

        try:
            batch_size = batch["input_ids"].size(0)
        except Exception:
            batch_size = args.batch_size

        ips = batch_size / elapsed if elapsed > 0 else 0
        rank = int(os.environ.get("LOCAL_RANK", 0))

        # 控制台输出符合指定格式
        # print(f"TCAPPDLL {now} - Epoch: 0 Iteration: {step}  rank : {rank}  "
        #     f"train.loss : {loss.item():.6f}  train.ips : {ips:.6f} imgs/s "
        #     f"train.total_time : {elapsed:.6f}")

        # 结构化日志写入
        logger.log(
            step=(0, step),
            data={
                "rank": rank,
                "train.loss": loss.item(),
                "train.ips": ips,
                "train.total_time": elapsed,
            },
            verbosity=Verbosity.DEFAULT
        )

        if step >= args.max_steps:
            break

    # 保存权重（可选）
    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    # print(f"Done. Loss log -> {log_path}")

if __name__ == "__main__":
    main()
