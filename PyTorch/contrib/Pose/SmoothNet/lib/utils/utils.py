# Adapted to tecorigin hardware
import os
from os import path as osp
import time
import yaml
import numpy as np
import torch
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

def create_logger(logdir):
    # os.makedirs(logdir, exist_ok=True)

    log_file = osp.join(logdir, f'log.json')


    logger = Logger(
        [
            StdOutBackend(Verbosity.DEFAULT),
            JSONStreamBackend(Verbosity.VERBOSE, log_file),
        ]
    )

    return logger


def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)


def prepare_output_dir(cfg, cfg_file):

    # ==== create logdir
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{cfg.EXP_NAME}'

    logdir = osp.join(cfg.OUTPUT_DIR, logdir)
    
    dir_num=0
    logdir_tmp=logdir

    while os.path.exists(logdir_tmp):
        logdir_tmp = logdir + str(dir_num)
        dir_num+=1
    
    logdir=logdir_tmp
    
    os.makedirs(logdir, exist_ok=True)
    #shutil.copy(src=cfg_file, dst=osp.join(cfg.OUTPUT_DIR, 'config.yaml'))

    cfg.LOGDIR = logdir

    # save config
    save_dict_to_yaml(cfg, osp.join(cfg.LOGDIR, 'config.yaml'))

    return cfg


def worker_init_fn(worker_id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

def slide_window_to_sequence(slide_window,window_step,window_size):
    if window_size == 1:
        return slide_window[:,0]
    N, W, C = slide_window.shape #[1440,8,51]
    output_len = (N - 1) * window_step + window_size
    slide_window = slide_window[:,None].permute(3,1,0,2)
    weight = torch.eye(window_size, device=slide_window.device, dtype=slide_window.dtype)[None,None,...].flip(2).contiguous() # [1,1,8,8]
    if 'int' in str(slide_window.dtype):
        sequence = torch.conv2d(slide_window.float(), weight.float(), padding=(window_size-1,0)) # [51,1,1447,1]
    else:
        sequence = torch.conv2d(slide_window, weight, padding=(window_size-1,0)) # [51,1,1447,1]
    sequence = sequence[:,0,:,0].permute(1,0)
    head_tail = 1 / torch.arange(1,window_size,1,device=slide_window.device)
    mean = torch.ones_like((sequence[:,0,None])) / window_size
    mean[0:window_size-1,0] = head_tail
    mean[1-window_size:, 0] = head_tail.flip(0)
    sequence = sequence * mean
    return sequence