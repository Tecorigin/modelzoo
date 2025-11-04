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
import sys
import os

# 添加项目根目录到 PYTHONPATH（即 ControlNet 所在目录）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import lightning_teco  # 注册 SDAA 加速器
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.model import create_model, load_state_dict
from argument import parse_args

class UnifiedLogger(pl.Callback):
    def __init__(self, logfile="sdaa.log"):
        super().__init__()
        self.start_time = None
        self.log_file = open(logfile, "a")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time

        loss = outputs['loss'] if isinstance(outputs, dict) and 'loss' in outputs else outputs
        loss_value = loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else float(loss)

        if isinstance(batch, dict) and 'jpg' in batch:
            batch_size = batch['jpg'].shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = len(batch)
        else:
            batch_size = batch.shape[0] if hasattr(batch, 'shape') else 1

        ips = batch_size / elapsed if elapsed > 0 else 0
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        log_str = (
            f"TCAPPDLL {now} - Epoch: {trainer.current_epoch} Iteration: {batch_idx} rank : 0 "
            f"train.loss : {loss_value:.8f} train.ips : {ips:.8f} imgs/s "
            f"train.loss : {loss_value:.8f} train.total_time : {elapsed:.8f}"
        )

        print(log_str)
        self.log_file.write(log_str + "\n")
        self.log_file.flush()

    def on_train_end(self, trainer, pl_module):
        self.log_file.close()

def main():
    args = parse_args()

    torch.cuda.empty_cache()

    model = create_model(args.config).cpu()
    model.load_state_dict(load_state_dict(args.ckpt, location='cpu'))
    model.learning_rate = args.lr
    model.sd_locked = True
    model.only_mid_control = False

    dataset = MyDataset(max_samples=200)
    dataloader = DataLoader(dataset, num_workers=2, batch_size=args.batchsize, shuffle=True)

    trainer = Trainer(
        accelerator='sdaa',
        devices=1,
        max_epochs=args.epoch,
        limit_train_batches=args.max_iter,
        precision="bf16",
        callbacks=[UnifiedLogger(logfile=args.logfile)]
    )

    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
