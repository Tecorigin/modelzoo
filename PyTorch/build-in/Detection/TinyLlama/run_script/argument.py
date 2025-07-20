#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Argument parser for ConvNeXt + SDAA quick experiments.

特点：
- 兼容单机 / 单卡快速调试
- 可选 SDAA / CUDA / CPU 设备
- 支持通过 --cfg-options 覆盖配置（仿 mmengine DictAction）
"""

import os
import argparse
from typing import Any, Dict, List


# --------- 兼容 mmengine 的 DictAction（若未安装 mmengine 则自定义一个） ----------
try:
    # 如果本地已安装 mmengine，可用其原生 DictAction
    from mmengine.config import DictAction  # type: ignore
except Exception:
    class DictAction(argparse.Action):
        """
        将形如 key=value 或 a.b.c=xxx 的字符串列表解析成嵌套 dict。
        用法示例:
            --cfg-options model.backbone.depth=12 optim.lr=0.001 seed=42
        """
        def __call__(self,
                     parser: argparse.ArgumentParser,
                     namespace: argparse.Namespace,
                     values: List[str],
                     option_string: str = None) -> None:
            cfg_dict: Dict[str, Any] = getattr(namespace, self.dest, {}) or {}
            for v in values:
                if '=' not in v:
                    raise argparse.ArgumentError(self, f"Expected key=value, got: {v}")
                key, value = v.split('=', 1)

                # 尝试把 value 转成 int / float / bool / None / list
                parsed: Any = value
                low = value.lower()
                if low in {'true', 'false'}:
                    parsed = (low == 'true')
                elif low in {'none', 'null'}:
                    parsed = None
                else:
                    # 尝试数字
                    try:
                        if '.' in value:
                            parsed = float(value)
                        else:
                            parsed = int(value)
                    except ValueError:
                        # 尝试列表：逗号分隔
                        if ',' in value:
                            parts = value.split(',')
                            casted = []
                            for p in parts:
                                p_l = p.lower()
                                if p_l in {'true', 'false'}:
                                    casted.append(p_l == 'true')
                                else:
                                    try:
                                        if '.' in p:
                                            casted.append(float(p))
                                        else:
                                            casted.append(int(p))
                                    except ValueError:
                                        casted.append(p)
                            parsed = casted
                        # 否则保持原字符串

                # 支持 a.b.c 递归
                cur = cfg_dict
                keys = key.split('.')
                for k in keys[:-1]:
                    if k not in cur or not isinstance(cur[k], dict):
                        cur[k] = {}
                    cur = cur[k]  # type: ignore
                cur[keys[-1]] = parsed
            setattr(namespace, self.dest, cfg_dict)


def parse_args():
    """Parse command-line arguments for ConvNeXt+SDAA quick experiments."""
    parser = argparse.ArgumentParser(
        description='Quick train / debug for ConvNeXt + SDAA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- 基本 ---
    parser.add_argument('--config', type=str,
                        help='MMEngine/MMDet style config path')
    parser.add_argument('--work-dir', default='./work_dirs',
                        help='Directory to save logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from given checkpoint (or "auto")')

    # --- 数据与训练 ---
    parser.add_argument('--data-path', default='data/teco-data/imagenet',
                        help='ImageNet root dir (must contain train/ and val/)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Global batch size (单进程情况下即每步 batch)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers per process')
    parser.add_argument('--device', default='sdaa',
                        choices=['sdaa', 'cuda', 'cpu'],
                        help='Device backend')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Stop after N iterations (debug)')

    # --- 训练选项 ---
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed-precision (torch.cuda.amp / sdAA amp)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation loop')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='Warmup steps (if scheduler supports)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Print log every N steps')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='Save checkpoint every N steps (0=disable)')

    # --- Scheduler  & 优化器额外配置 ---
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'linear', 'constant'],
                        help='LR scheduler type')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Min LR for cosine/linear decay')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0,
                        help='Gradient norm clipping (0/负数=禁用)')

    # --- dataloader 性能调节 ---
    parser.add_argument('--pin-mem', action='store_true',
                        help='Use pinned-memory dataloader')
    parser.add_argument('--persistent-workers', action='store_true',
                        help='Use persistent dataloader workers')

    # --- 配置覆写 ---
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Override config keys, e.g. model.backbone.depth=12 optim.lr=0.001')

    # --- 结果可视化 / 监控 ---
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging (需提前 login)')
    parser.add_argument('--project', type=str, default='convnext-sdaa',
                        help='W&B project name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='W&B run name')

    # --- 预留（分布式） ---
    parser.add_argument('--local_rank', type=int, default=0,
                        help='(For DDP launcher compatibility)')

    args = parser.parse_args()

    # 若想在单进程环境中与部分库兼容 local_rank
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


if __name__ == "__main__":
    a = parse_args()
    print("Parsed args:")
    for k, v in vars(a).items():
        print(f"  {k}: {v}")
