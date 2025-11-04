import torch
import torch_sdaa
import visdom
from tqdm import trange
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
class Train(object):
    def __init__(self, model, data_loader, optimizer, criterion, lr, wd, batch_size, vis, device):
        super(Train, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr = lr
        self.wd = wd
        self.bs = batch_size
        self.vis = None
        self.device = device
        self.losses = []
        # 加入混合精度计算
        self.scaler = torch.sdaa.amp.GradScaler()
        self.amp_bool = True  # 是否使用混合精度计算，默认否
        if vis:
            self.vis = visdom.Visdom()

            self.loss_window = self.vis.line(X=torch.zeros((1,)).cpu(),
                    Y=torch.zeros((1)).cpu(),
                    opts=dict(xlabel='minibatches',
                    ylabel='Loss',
                    title='Training Loss',
                    legend=['Loss']))

        self.iterations = 0
        self.iterations_list = []
        self.save_path = "./output_plot/"
    def forward(self, logger):
        # start_time = time.time()
        self.model.train()
        # TODO adjust learning rate

        total_loss = 0
        pbar = trange(len(self.data_loader.dataset), desc='Training ')

        for batch_idx, (x, yt) in enumerate(self.data_loader):
            # # x = x.cuda(non_blocking=True)
            # # yt = yt.cuda(non_blocking=True)
            # x = x.sdaa(non_blocking=True)
            # yt = yt.sdaa(non_blocking=True)
            # input_var = Variable(x)
            # target_var = Variable(yt)
            if self.amp_bool:
                input_var = x.to(self.device)
                target_var = yt.to(self.device)
                # print("正在使用自动混合精度计算")
                with torch.sdaa.amp.autocast():  # 开启AMP环境
                    y = self.model(input_var)
                    loss = self.criterion(y, target_var)
                    self.losses.append(loss.item())
                self.iterations_list.append(self.iterations)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()  # loss缩放并反向转播
                self.scaler.step(self.optimizer)  # 参数更新
                self.scaler.update()  # 基于动态Loss Scale更新loss_scaling系数
            else:
                input_var = x.to(self.device)
                target_var = yt.to(self.device)
                # print("正在不使用自动混合精度计算")
                # compute output
                y = self.model(input_var)
                loss = self.criterion(y, target_var)
                self.losses.append(loss.item())
                self.iterations_list.append(self.iterations)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            logger.write('\n{:.6f}'.format(loss.item()))
            if batch_idx % 10 == 0:
                # Update tqdm bar
                if (batch_idx*self.bs + 10*len(x)) <= len(self.data_loader.dataset):
                    pbar.update(10 * len(x))
                else:
                    pbar.update(len(self.data_loader.dataset) - int(batch_idx*self.bs))

            # Display plot using visdom
            if self.vis:
                self.vis.line(
                        X=torch.ones((1)).cpu() * self.iterations,
                        Y=loss.data.cpu(),
                        win=self.loss_window,
                        update='append')

            self.iterations += 1

        pbar.close()

        return total_loss*self.bs/len(self.data_loader.dataset)
        # return total_loss*self.bs/len(self.data_loader.dataset),self.losses
