# Adapted to tecorigin hardware。

import torch
import torch.nn as nn


from src.core import register
import src.misc.dist as dist


__all__ = ['GradScaler']

# [修改] 将 amp.grad_scaler.GradScaler 改为 torch.sdaa.amp.grad_scaler.GradScaler
GradScaler = register(torch.sdaa.amp.grad_scaler.GradScaler)
