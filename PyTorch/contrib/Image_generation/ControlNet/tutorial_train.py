from share import *
from pytorch_lightning import Trainer
import lightning_teco  # 必须 import 触发注册
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.model import create_model, load_state_dict
import torch
torch.cuda.empty_cache()

# ===== 配置 =====
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 1
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# ===== 创建模型 =====
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# ===== 数据加载 =====
dataset = MyDataset(max_samples=200)
dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True)

# ===== 自定义日志回调：每 iter 写入 sdaa.log =====
import pytorch_lightning as pl
import time
import datetime

class UnifiedLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.log_file = open("sdaa.log", "a")  # ✅ 打开 sdaa.log 追加写入

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time

        # ✅ 保证从 outputs 中安全提取 loss
        loss = outputs['loss'] if isinstance(outputs, dict) and 'loss' in outputs else outputs
        loss_value = loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else float(loss)

        # ✅ 获取 batch 大小
        if isinstance(batch, dict) and 'jpg' in batch:
            batch_size = batch['jpg'].shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = len(batch)
        else:
            batch_size = batch.shape[0] if hasattr(batch, 'shape') else 1

        # ✅ 计算 IPS
        ips = batch_size / elapsed if elapsed > 0 else 0
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # ✅ 构建日志字符串
        log_str = (
            f"TCAPPDLL {now} - Epoch: {trainer.current_epoch} Iteration: {batch_idx} rank : 0 "
            f"train.loss : {loss_value:.8f} train.ips : {ips:.8f} imgs/s "
            f"train.loss : {loss_value:.8f} train.total_time : {elapsed:.8f}"
        )

        # ✅ 打印 & 写入日志
        print(log_str)
        self.log_file.write(log_str + "\n")
        self.log_file.flush()

    def on_train_end(self, trainer, pl_module):
        self.log_file.close()  # ✅ 训练结束关闭文件

# ===== 启动训练器（使用 SDAA）=====
trainer = Trainer(
    accelerator='sdaa',
    devices=1,
    max_epochs=1,
    limit_train_batches=100,
    precision="bf16",
    callbacks=[UnifiedLogger()]
)

# ===== 启动训练 =====
trainer.fit(model, dataloader)
