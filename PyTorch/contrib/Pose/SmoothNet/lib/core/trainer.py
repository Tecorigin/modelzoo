import time
import os
import copy
import torch
import logging
import torch_sdaa
from lib.core.loss import *
from progress.bar import Bar

from lib.utils.eval_metrics import *
from lib.utils.geometry_utils import *
from tcap_dllogger import Logger, Verbosity

class Trainer():  # merge

    def __init__(self,
                 cfg,
                 train_dataloader,
                 test_dataloader,
                 model,
                 loss,
                 optimizer,
                 logger:Logger,
                 start_epoch=0):
        super().__init__()
        self.cfg = cfg

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.logger = logger

        # Training parameters
        self.logdir = cfg.LOGDIR

        self.start_epoch = start_epoch
        self.end_epoch = cfg.TRAIN.EPOCH
        self.epoch = 0

        self.train_global_step = 0
        self.valid_global_step = 0
        self.device = cfg.DEVICE
        self.resume = cfg.TRAIN.RESUME
        self.lr = cfg.TRAIN.LR



    def run(self):
        for epoch_num in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch_num
            
            self.train()
            if self.cfg.TRAIN.VALIDATE:
                performance = self.evaluate()
                self.save_model(performance, epoch_num)
            else:
                self.save_model(None, epoch_num)

            # Decay learning rate exponentially
            lr_decay = self.cfg.TRAIN.LRDECAY
            self.lr *= lr_decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay

        if not self.cfg.TRAIN.VALIDATE:
            performance = self.evaluate()

    def tcap_logger(self, data, verbosity=Verbosity.DEFAULT):
        self.logger.log(step = (self.epoch, self.train_global_step), data = data, verbosity=verbosity)
    
    def train(self):

        self.model.train()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        start = time.time()
        if self.cfg.AMP:
            scaler = torch_sdaa.amp.GradScaler() 
        training_iter=min([len(self.train_dataloader[i]) for i in range(len(self.train_dataloader))])

        train_dataloader=copy.deepcopy(self.train_dataloader)
        for dataloader_index in range(len(train_dataloader)):
            train_dataloader[dataloader_index]=iter(train_dataloader[dataloader_index])

        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}',
                  fill='*',
                  max=training_iter*len(train_dataloader))

        for iter_index in range(training_iter):
            for data_index in range(len(train_dataloader)):
                present_train_dataloader=train_dataloader[data_index]

                data=next(present_train_dataloader)

                data_pred = data["pred"].to(self.device)
                data_gt = data["gt"].to(self.device)

                timer['data'] = time.time() - start
                start = time.time()

                self.optimizer.zero_grad()

                data_pred=data_pred.permute(0,2,1)
                if self.cfg.AMP:
                    with torch_sdaa.amp.autocast(): 
                        denoised_pos = self.model(data_pred)
                        denoised_pos=denoised_pos.permute(0,2,1)
                        timer['forward'] = time.time() - start
                        start = time.time()
                        loss_total = self.loss(denoised_pos, data_gt)
                        timer['loss'] = time.time() - start
                        start = time.time()
                    scaler.scale(loss_total).backward()    # loss缩放并反向转播
                    scaler.step(self.optimizer)    # 参数更新
                    scaler.update()    # 基于动态Loss Scale更新loss_scaling系数
                else:
                    denoised_pos = self.model(data_pred)
                    denoised_pos=denoised_pos.permute(0,2,1)
                    timer['forward'] = time.time() - start
                    start = time.time()
                    loss_total = self.loss(denoised_pos, data_gt)
                    timer['loss'] = time.time() - start
                    start = time.time()
                    loss_total.backward()
                    self.optimizer.step()
                timer['backward'] = time.time() - start
                timer['batch'] = timer['data'] + timer['forward'] + timer[
                    'loss'] + timer['backward']

                log_data = {'train.loss': loss_total.item(),
                            'train.lr': self.lr,
                            'train.time.data': timer['data'],
                            'train.time.forward': timer['forward'],
                            'train.time.loss': timer['loss'],
                            'train.time.backward': timer['backward'],
                            'train.time.batch': timer['batch'],
                            'train.ips': denoised_pos.shape[0] / timer['batch']}
                self.tcap_logger(log_data)
                
                self.train_global_step += 1
                if torch.isnan(loss_total):
                    exit('Nan value in loss, exiting!...')

                # bar.suffix = f"loss: {loss_total.item()}"
                bar.next()


                

    def evaluate_3d(self,dataset_index,dataset,estimator):

        eval_dict = evaluate_smoothnet_3D(self.model, self.test_dataloader[dataset_index],
                                          self.device,dataset,estimator, self.cfg)
        log_dict = {}
        for k,v in eval_dict.items():
            log_dict[f'val.{k}'] = v
        self.tcap_logger(log_dict)

        return eval_dict

    def evaluate_smpl(self,dataset_index,dataset):
        eval_dict = evaluate_smoothnet_smpl(self.model, self.test_dataloader[dataset_index],
                                            self.device, self.cfg,dataset)
        
        log_dict = {}
        for k,v in eval_dict.items():
            log_dict[f'val.{k}'] = v
        self.tcap_logger(log_dict)

        return eval_dict

    def evaluate_2d(self,dataset_index,dataset):

        eval_dict = evaluate_smoothnet_2D(self.model, self.test_dataloader[dataset_index],
                                          self.device, self.cfg,dataset)

        log_dict = {}
        for k,v in eval_dict.items():
            log_dict[f'val.{k}'] = v
        self.tcap_logger(log_dict)

        return eval_dict

    def evaluate(self):

        self.model.eval()

        performance=[]
        all_dataset=self.cfg.DATASET_NAME.split(",")
        all_body_representation=self.cfg.BODY_REPRESENTATION.split(",")
        all_estimator=self.cfg.ESTIMATOR.split(",")

        for dataset_index in range(len(all_dataset)):
            present_representation= all_body_representation[dataset_index]
            present_dataset=all_dataset[dataset_index]
            present_estimator=all_estimator[dataset_index]
            print("=======================================================")
            print("evaluate on dataset: "+present_dataset+", estimator: "+present_estimator+", body representation: "+present_representation)
            
            if present_representation == "3D":
                performance.append(self.evaluate_3d(dataset_index,present_dataset,present_estimator))

            elif present_representation == "smpl":
                performance.append(self.evaluate_smpl(dataset_index,present_dataset))

            elif present_representation == "2D":
                performance.append(self.evaluate_2d(dataset_index,present_dataset))

        return performance

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'performance': performance,
            'optimizer': self.optimizer.state_dict()
        }

        filename = os.path.join(self.logdir, f'epoch_{epoch}.pth.tar')
        torch.save(save_dict, filename)
