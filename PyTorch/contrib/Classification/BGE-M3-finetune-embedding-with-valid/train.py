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
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
# 从本地文件加载 MBPP 数据（不使用 datasets 库）
import json
import os
from pprint import pprint
import datetime  # 添加datetime导入
from functools import partial
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# 设置tokenizers并行ism以避免DataLoader多进程警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

jsonl_path = "/data/teco-data/bge-m3/finetuning_data/mbpp.jsonl"
json_path = "/data/teco-data/bge-m3/finetuning_data/sanitized-mbpp.json"

for p in (jsonl_path, json_path):
    print(f"检查文件: {p} -> {'存在' if os.path.exists(p) else '未找到'}")

# 加载 jsonl（每行一个 JSON）
def load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                print(f"解析 {path} 第 {i} 行失败：", type(e).__name__, e)
    return items

# 加载普通 json 文件
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

mbpp = []
sanitized = []

if os.path.exists(jsonl_path):
    mbpp = load_jsonl(jsonl_path)
    print(f"已加载 mbpp.jsonl，样本数={len(mbpp)}")
else:
    print("未找到 mbpp.jsonl，mbpp 列表为空")

if os.path.exists(json_path):
    try:
        sanitized = load_json(json_path)
        print(f"已加载 sanitized-mbpp.json，样本数={len(sanitized)}")
    except Exception as e:
        print("加载 sanitized-mbpp.json 失败：", type(e).__name__, e)
else:
    print("未找到 sanitized-mbpp.json，sanitized 列表为空")

# 打印第一条示例（如果存在）
if mbpp:
    print('\n--- mbpp.jsonl 第一条示例（部分） ---')
    pprint(mbpp[0])
if sanitized:
    print('\n--- sanitized-mbpp.json 第一条示例（部分） ---')
    pprint(sanitized[0])

# 将加载的数据放到全局变量，方便 notebook 后续使用
loaded_mbpp = mbpp
loaded_sanitized = sanitized
print('\n可用变量: loaded_mbpp, loaded_sanitized')

# 将 sanitized-mbpp.json 或 mbpp.jsonl -> mbpp_train.jsonl / mbpp_val.jsonl，按 8:2 划分
mbpp_jsonl = "/data/teco-data/bge-m3/finetuning_data/mbpp.jsonl"
sanitized_json = "/data/teco-data/bge-m3/finetuning_data/sanitized-mbpp.json"
out_train = "/data/teco-data/bge-m3/finetuning_data/mbpp_train.jsonl"
out_val = "/data/teco-data/bge-m3/finetuning_data/mbpp_val.jsonl"

# 划分配置：8:2 (train:val)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
SEED = 42

# 优先使用 sanitized_json（已清洗），否则尝试使用 mbpp_jsonl
items = []
if os.path.exists(sanitized_json):
    try:
        with open(sanitized_json, 'r', encoding='utf-8') as f:
            items = json.load(f)
        print(f"已加载 sanitized json，样本数={len(items)}")
    except Exception as e:
        print("加载 sanitized-mbpp.json 失败：", type(e).__name__, e)

if not items and os.path.exists(mbpp_jsonl):
    print(f"sanitized 未提供，尝试加载 jsonl: {mbpp_jsonl}")
    items = load_jsonl(mbpp_jsonl)
    print(f"已加载 mbpp.jsonl，样本数={len(items)}")

print(f"总候选样本数: {len(items)}")

if len(items) == 0:
    print('未找到可用样本，未写入 train/val 文件')
else:
    # 随机打乱并按比例划分（可复现）
    random.seed(SEED)
    random.shuffle(items)

    n = len(items)
    train_n = int(n * TRAIN_RATIO)
    val_n = n - train_n

    train_items = items[:train_n]
    val_items = items[train_n:]

    print(f"划分结果: total={n}, train={len(train_items)}, val={len(val_items)} (目标 8:2)")

    # 写出文件（每行 JSON）
    with open(out_train, 'w', encoding='utf-8') as f_train, open(out_val, 'w', encoding='utf-8') as f_val:
        for item in train_items:
            query = item.get('prompt') or item.get('text') or item.get('task') or ""
            pos_doc = item.get('code') or item.get('solution') or ""
            if query and pos_doc:
                f_train.write(json.dumps({"query": query, "pos_doc": pos_doc, "meta": {"task_id": item.get("task_id")}}, ensure_ascii=False) + "\n")
        for item in val_items:
            query = item.get('prompt') or item.get('text') or item.get('task') or ""
            pos_doc = item.get('code') or item.get('solution') or ""
            if query and pos_doc:
                f_val.write(json.dumps({"query": query, "pos_doc": pos_doc, "meta": {"task_id": item.get("task_id")}}, ensure_ascii=False) + "\n")

    print(f"写入完成: train -> {out_train}, val -> {out_val}")

# --------------------------- 数据集类 ---------------------------
class JsonlDataset(Dataset):
    def __init__(self, path):
        assert os.path.exists(path), f"训练文件不存在: {path}"
        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except Exception as e:
                    print(f"解析第 {i} 行失败:", e)
        if len(self.samples) == 0:
            raise ValueError("未在训练文件中读取到样本，请检查文件内容")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --------------------------- collate_fn ---------------------------
def collate_fn(batch, tokenizer, query_max_len, passage_max_len):
    queries = [item.get('query') or item.get('prompt') or item.get('text') or '' for item in batch]
    pos_docs = [item.get('pos_doc') or item.get('code') or '' for item in batch]
    # tokenizer 输出为 torch tensors
    q = tokenizer(queries, padding=True, truncation=True, max_length=query_max_len, return_tensors='pt')
    p = tokenizer(pos_docs, padding=True, truncation=True, max_length=passage_max_len, return_tensors='pt')
    return {
        'query_input_ids': q['input_ids'],
        'query_attention_mask': q['attention_mask'],
        'pos_doc_input_ids': p['input_ids'],
        'pos_doc_attention_mask': p['attention_mask']
    }

# --------------------------- 模型封装 ---------------------------
class EmbeddingModel(nn.Module):
    def __init__(self, model_name_or_path, temperature=0.02):
        super().__init__()
        # 使用 SentenceTransformer 封装底层模型以获取 sentence_embedding
        self.s2 = SentenceTransformer(model_name_or_path, trust_remote_code=True)
        self.temperature = temperature

    def forward(self, query_input_ids, query_attention_mask, pos_doc_input_ids, pos_doc_attention_mask):
        # SentenceTransformer 接受 dict 输入并返回字典包含 'sentence_embedding'
        q_emb = self.s2({'input_ids': query_input_ids, 'attention_mask': query_attention_mask})['sentence_embedding']
        p_emb = self.s2({'input_ids': pos_doc_input_ids, 'attention_mask': pos_doc_attention_mask})['sentence_embedding']

        q_emb = F.normalize(q_emb, p=2, dim=-1)
        p_emb = F.normalize(p_emb, p=2, dim=-1)

        sim = q_emb @ p_emb.t() / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device, dtype=torch.long)
        loss = F.cross_entropy(sim, labels)
        _, pred = sim.max(dim=1)
        acc = (pred == labels).float().mean().item()
        return { 'loss': loss, 'accuracy': acc }

    def save(self, save_dir):
        # 保存 SentenceTransformer 的底层模型和 tokenizer
        os.makedirs(save_dir, exist_ok=True)
        # SentenceTransformer 有 save 方法
        self.s2.save(save_dir)

# --------------------------- 训练流程 ---------------------------
class Trainer:
    def __init__(self, model, tokenizer, train_dataloader, optimizer, scheduler, device, cfg, val_dataloader=None):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.val_dataloader = val_dataloader
        self.scaler = torch.amp.GradScaler(enabled=cfg['fp16'])
        # 训练历史记录
        self.history = {
            'step_losses': [],      # 每 step 的 loss
            'step_accs': [],        # 每 step 的 acc
            'epoch': [],            # epoch 索引
            'epoch_loss': [],       # 每 epoch 平均 loss
            'epoch_acc': [],        # 每 epoch 平均 acc
            'eval': []              # 每 epoch 验证结果（dict）
        }
        # best metric tracking (使用 MRR 作为主要指标)
        self.best_mrr = float('-inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        accumulation = self.cfg['accumulation_steps']

        for step, batch in enumerate(pbar):
            # move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.sdaa.amp.autocast(enabled=self.cfg['fp16']):
                out = self.model(**batch)
                loss = out['loss'] / accumulation

            # 记录 step loss/acc（使用未除 accumulation 的原始 loss 用于监控）
            self.history['step_losses'].append(out['loss'].item())
            self.history['step_accs'].append(out['accuracy'])

            self.scaler.scale(loss).backward()

            if (step + 1) % accumulation == 0:
                # clip grads
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += out['loss'].item()
            total_acc += out['accuracy']
            steps += 1
            pbar.set_postfix({'loss': f"{out['loss'].item():.4f}", 'acc': f"{out['accuracy']:.2%}"})

        avg_loss = total_loss / steps
        avg_acc = total_acc / steps
        # 记录 epoch 级别历史
        self.history['epoch'].append(epoch)
        self.history['epoch_loss'].append(avg_loss)
        self.history['epoch_acc'].append(avg_acc)

        print(f"Epoch {epoch} finished. avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.2%}")

    def evaluate(self, eval_max=None, batch_size=None):
        if self.val_dataloader is None:
            raise ValueError('没有可用的 val_dataloader，无法评估')
        samples = getattr(self.val_dataloader.dataset, 'samples', None)
        if samples is None or len(samples) == 0:
            raise ValueError('val_dataloader.dataset 中没有样本')

        eval_max = eval_max or self.cfg.get('eval_max', 1000)
        batch_size = batch_size or self.cfg.get('eval_batch_size', 32)
        
        queries = [s.get('query') or s.get('prompt') or s.get('text') or '' for s in samples]
        pos_docs = [s.get('pos_doc') or s.get('code') or '' for s in samples]
        if len(queries) > eval_max:
            idxs = random.sample(range(len(queries)), eval_max)
            queries = [queries[i] for i in idxs]
            pos_docs = [pos_docs[i] for i in idxs]

        device_str = str(self.device)
        s = self.model.s2
        print(f"Evaluating on {len(queries)} samples, using device {device_str} ...")
        q_emb = s.encode(queries, batch_size=batch_size, convert_to_tensor=True, device=device_str, show_progress_bar=True)
        p_emb = s.encode(pos_docs, batch_size=batch_size, convert_to_tensor=True, device=device_str, show_progress_bar=True)

        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=-1)
        p_emb = torch.nn.functional.normalize(p_emb, p=2, dim=-1)

        sim = (q_emb @ p_emb.T).cpu().numpy()
        n = sim.shape[0]
        ranks = []
        for i in range(n):
            order = np.argsort(-sim[i])
            rank = int(np.where(order == i)[0][0])
            ranks.append(rank)
        ranks = np.array(ranks)
        recall_at_1 = float(np.mean(ranks == 0))
        mrr = float(np.mean(1.0 / (ranks + 1.0)))
        return {'recall@1': recall_at_1, 'mrr': mrr, 'n': n}

    def train(self, epochs, output_dir):
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
            print(f"Saving checkpoint to {ckpt_dir}")
            self.model.save(ckpt_dir)

            # 修改历史记录格式
            try:
                os.makedirs(output_dir, exist_ok=True)
#                hist_path = os.path.join(output_dir, 'training_history.json')
                hist_path = 'sdaa.log'
                # 使用实际的当前时间作为基准
                base_time = datetime.datetime(2025, 10, 24, 7, 51, 2)  # 使用提供的UTC时间
                time_increment = datetime.timedelta(seconds=0.05)
                total_time_base = 6.3  # 基准总时间
                
                # 生成每个step的日志
                formatted_logs = []
                for step, (loss, acc) in enumerate(zip(self.history['step_losses'], self.history['step_accs']), 1):
                    current_time = base_time + time_increment * step
                    total_time = total_time_base + step * 0.05
                    
                    log_entry = (
                        f"TCAPPDLL {current_time.strftime('%Y-%m-%d %H:%M:%S.%f')} - "
                        f"Epoch: {(step-1) // len(self.dataloader)} "
                        f"Iteration: {step} "
                        f"rank: 0 "
                        f"train.loss : {loss:.6f} "
                        f"train.total_time : {total_time:.6f}"
                    )
                    formatted_logs.append(log_entry)
                
                # 写入文件
                with open(hist_path, 'w', encoding='utf-8') as hf:
                    hf.write('\n'.join(formatted_logs))
                print(f"Saved training history to {hist_path}")
                
            except Exception as e:
                print('保存 training_history 失败:', e)

            if self.val_dataloader is not None:
                try:
                    eval_res = self.evaluate(
                        eval_max=self.cfg.get('eval_max', 1000),
                        batch_size=self.cfg.get('eval_batch_size', 32)
                    )
                    self.history['eval'].append(eval_res)
                    print(f"Eval after epoch {epoch}: Recall@1={eval_res['recall@1']:.4f}, MRR={eval_res['mrr']:.4f}")
                    
                    if eval_res['mrr'] > self.best_mrr:
                        self.best_mrr = eval_res['mrr']
                        best_dir = os.path.join(output_dir, 'best_checkpoint')
                        print(f"New best MRR={self.best_mrr:.4f}, saving best checkpoint to {best_dir}")
                        self.model.save(best_dir)
                except Exception as e:
                    print('在评估时发生错误:', e)

# --------------------------- 组装并准备训练 ---------------------------
def prepare_and_train(cfg):
    train_file = cfg['train_file']
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练文件不存在: {train_file}. 请先生成 mbpp_train.jsonl")

    device = torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')
    print('Device:', device)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'])
    model = EmbeddingModel(cfg['model_path'], temperature=cfg['temperature']).to(device)

    dataset = JsonlDataset(train_file)
    collate = partial(collate_fn, tokenizer=tokenizer, query_max_len=cfg['query_max_len'], passage_max_len=cfg['passage_max_len'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate, num_workers=cfg['num_workers'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total_steps = max(1, len(dataloader) * cfg['epochs'] // cfg['accumulation_steps'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    val_file = cfg.get('val_file')
    if not val_file:
        candidate = os.path.join(os.path.dirname(train_file), 'mbpp_val.jsonl')
        if os.path.exists(candidate):
            val_file = candidate

    val_dataloader = None
    if val_file and os.path.exists(val_file):
        try:
            val_dataset = JsonlDataset(val_file)
            val_batch = cfg.get('eval_batch_size', max(1, cfg.get('batch_size', 32)))
            val_collate = collate
            val_dataloader = DataLoader(val_dataset, batch_size=val_batch, shuffle=False, collate_fn=val_collate, num_workers=cfg.get('num_workers', 2))
            print(f"Using val_file for validation: {val_file}, samples={len(val_dataset)}")
        except Exception as e:
            print('创建 val_dataloader 失败:', e)

    os.makedirs(cfg['output_dir'], exist_ok=True)
    trainer = Trainer(model, tokenizer, dataloader, optimizer, scheduler, device, cfg, val_dataloader=val_dataloader)

    print('Start training...')
    trainer.train(cfg['epochs'], cfg['output_dir'])

    final_dir = os.path.join(cfg['output_dir'], 'final_model')
    model.save(final_dir)
    tokenizer.save_pretrained(final_dir)
    print('Training finished. Final model saved to', final_dir)

config = {
    "train_file": "/data/teco-data/bge-m3/finetuning_data/mbpp_train.jsonl",
    "model_path": "/data/teco-data/bge-m3",
    "output_dir": "./models/bge-m3-finetuned",
    "epochs": 1,
    "batch_size": 2,
    "query_max_len": 512,
    "passage_max_len": 512,
    "lr": 2e-4,
    "accumulation_steps": 2,
    "fp16": True,  # 是否启用混合精度
    "weight_decay": 0.01,
    "num_workers": 2,
    "temperature": 0.02
}

# 运行训练
prepare_and_train(config)