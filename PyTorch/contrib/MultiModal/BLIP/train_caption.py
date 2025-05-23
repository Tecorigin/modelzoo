'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import sys
import time
import torch
import torch_sdaa
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
from tqdm import tqdm
import time
def train(model, data_loader, optimizer, epoch, device,model_train_log_file,rank,args):
    # train
    model.train()  
    #开始自动混合精度训练
    if args.is_use_amp:
        scaler = torch.sdaa.amp.GradScaler()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50
    bsz = config['batch_size']
    loss_list = []
    start_time = time.time()
    
    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.is_use_amp:
            with torch.sdaa.amp.autocast():
                image = image.to(device)       
                loss = model(image, caption)      
            print("loss:",loss.item())
            if rank == 0:
                model_train_log_file.write(f"TrainLoss {loss.item()}\n")
                model_train_log_file.flush()
            loss_list.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            #loss.backward()
            scaler.step(optimizer)
            #optimizer.step()    
            scaler.update()
        else:
            image = image.to(device)       
            loss = model(image, caption)      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if time.time()-start_time > 7200:
            break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())    
    if rank ==0:
        model_train_log_file.write(f"AveragedStats {metric_logger.global_avg()}\n")
        model_train_log_file.flush() 
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)       
        
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    print("args:",args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module    
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    #设置log文件保存模型训练过程的数据
    model_train_log_path = './scripts/train_sdaa_3rd.log'
    model_train_log_file = open(model_train_log_path, 'w')
    for epoch in range(0, config['max_epoch']):
        time1 = time.time()
        if time1 - start_time > args.training_time*3600:
            sys.exit(1)
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device,model_train_log_file,rank,args) 
        '''val_result = evaluate(model_without_ddp, val_loader, device, config)  
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id') '''       
  
        '''test_result = evaluate(model_without_ddp, test_loader, device, config)  
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')  '''

        '''if utils.is_main_process():   
            coco_val = coco_caption_eval(config['coco_gt_root'],val_result_file,'val')
            coco_test = coco_caption_eval(config['coco_gt_root'],test_result_file,'test')
            
            if args.evaluate:            
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
            else:             
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

                if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                    best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                    best_epoch = epoch                
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
                    
        if args.evaluate: 
            break'''
        dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='sdaa')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes') 
    parser.add_argument('--training_time', default=6, type=float, help='hours to train')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--is_use_amp', default=True, type=bool)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        yaml = yaml.YAML(typ='rt')  # 使用 'rt' 类型读取文件
        config = yaml.load(file)
    #config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)