import datetime
import json
import time
import os
import sys
import argparse
import numpy as np
import torch
import torch_sdaa
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.clip import CLIP

from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import ClipDataset, dataset_collate
from utils.utils import (get_configs, get_lr_scheduler, set_optimizer_lr, show_config)
from utils.utils_fit import fit_one_epoch
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity


json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "test.json"),
    ]
)
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})

def setup_distributed(local_rank):
    device = torch.device(f"sdaa:{local_rank}")
    torch.sdaa.set_device(device)
    torch.distributed.init_process_group(backend="tccl", init_method="env://")
    return device

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    parser = argparse.ArgumentParser()
    print("start")

    parser.add_argument("--model_name", default=None, type=str, required=True)

    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--ckpt_path",
                            default=None,
                            type=str,
                            help="The checkpoint file from pretraining")
    parser.add_argument("--bert_ckpt_path",
                        default=None,
                        type=str,
                        help="The bert checkpoint file from pretraining")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    # parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--distributed", action='store_true', help="Whether to run training.")
    parser.add_argument("--epoch", default=20, type=int)
    parser.add_argument("--datasets_path",
                        default=None,
                        type=str)
    parser.add_argument('--device', required=True, default='sdaa', type=str,
                        help='which device to use. cpu, cuda, sdaa optional, cpu default')
    parser.add_argument("--nproc_per_node", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--amp",
                        default=False,
                        action='store_true',
                        help="Mixed precision training")

    print(parser)
    print(sys.argv)
    args = parser.parse_args()

    model_path = args.ckpt_path
    if "16" in model_path:
        phi = "VIT-B-16-cn"
    elif "32" in model_path:
        phi = "VIT-B-32-cn"
    else:
        assert False, "Unsupported model type in the model path. Please include either '16' or '32' in the path."

    bert_ckpt = args.bert_ckpt_path

    SDAA = True
    distributed = args.distributed
    fp16 = args.amp
    batch_size = args.batch_size
    Init_Epoch = 0
    Epoch = args.epoch
    world_size = args.nproc_per_node
    Init_lr = args.learning_rate
    Min_lr = Init_lr * 0.01


    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2

    lr_decay_type = 'cos'

    save_period = 1
    save_dir = args.output_dir

    eval_flag = True
    eval_period = 1

    num_workers = 1

    datasets_path = args.datasets_path
    datasets_train_json_path = os.path.join(datasets_path, "cn_train.json")
    datasets_val_json_path = os.path.join(datasets_path, "cn_val.json")
    print("train_path",datasets_train_json_path)
    print("val_path",datasets_val_json_path)

    datasets_random = True

    if distributed:
        device = setup_distributed(local_rank)
    else:
        device = torch.device('sdaa')
        local_rank = 0
        rank = 0

    config = get_configs(phi,bert_ckpt)
    model = CLIP(**config)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, None)
    else:
        loss_history = None

    if fp16:
        # from torch.sdaa.amp import GradScaler as GradScaler
        scaler = torch_sdaa.amp.GradScaler()
    else:
        scaler = None
    print("&&&&&&&&&&&&&&&&&&&&&&&&&", device)

    model = model.to(device)
    if SDAA:
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
        else:
            model = torch.nn.DataParallel(model)
            model.to(device)

    train_lines = json.load(open(datasets_train_json_path, mode='r', encoding='utf-8'))
    val_lines = json.load(open(datasets_val_json_path, mode='r', encoding='utf-8'))

    num_train = len(train_lines)
    num_val = len(val_lines)
    print('num_val',num_val)

    if local_rank == 0:
        show_config(
            model_path=model_path, phi=phi,
            Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    if True:
        nbs = 64
        lr_limit_max = 1e-5
        lr_limit_min = 3e-6
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = ClipDataset([config['input_resolution'], config['input_resolution']], train_lines,
                                    datasets_path, random=datasets_random)
        val_dataset = ClipDataset([config['input_resolution'], config['input_resolution']], val_lines, datasets_path,
                                  random=False)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // world_size
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        dataload_time = time.time()
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=False, collate_fn=dataset_collate, sampler=val_sampler)
        dataload_timeend = time.time() - dataload_time


        if local_rank == 0:
            eval_dataset = ClipDataset([config['input_resolution'], config['input_resolution']], val_lines,
                                       datasets_path, random=False)
            gen_eval = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, drop_last=False, collate_fn=dataset_collate, sampler=None)
            eval_callback = EvalCallback(model, gen_eval, log_dir, SDAA, eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        for epoch in range(Init_Epoch, Epoch):
            json_logger.log(
                step = [epoch],
                data = {
                "train.data_time": dataload_timeend,
            },
            verbosity = Verbosity.DEFAULT,
            )
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, Epoch, SDAA, fp16, scaler, save_period, save_dir, json_logger,distributed, local_rank)

        if local_rank == 0:
            loss_history.writer.close()


if __name__ == "__main__":
    main()