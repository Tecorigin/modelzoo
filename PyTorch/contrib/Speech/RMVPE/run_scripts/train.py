import importlib
import os
import warnings
import sys
from pathlib import Path


root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))


import argparse
import numpy as np
from tqdm import tqdm
from tcap_dllogger import Logger, JSONStreamBackend, Verbosity
logger = Logger(
    [
        JSONStreamBackend(Verbosity.VERBOSE, "tcap_json.log"),
    ]
)

try:
    import torch_sdaa
except ImportError:
    pass
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src import MIR1K, E2E0, cycle, summary, SAMPLE_RATE, bce
from evaluate import evaluate

# 设置IP:PORT，框架启动TCP Store为ProcessGroup服务
os.environ['MASTER_ADDR'] = 'localhost' # 设置IP
os.environ['MASTER_PORT'] = '29503'     # 设置端口号

def train(rank, logdir, hop_length, learning_rate,
          batch_size, validation_interval, clip_grad_norm,
          world_size, iterations, learning_rate_decay_steps,
          learning_rate_decay_rate, resume_iteration, device):

    device = torch.device(f"sdaa:{rank}")
    torch.sdaa.set_device(device)
    # init DDP
    dist.init_process_group('tccl', rank=rank, world_size=world_size)
    
    train_dataset = MIR1K('Hybrid', hop_length, ['train'], whole_audio=False, use_aug=True, rank=rank)
    validation_dataset = MIR1K('Hybrid', hop_length, ['test'], whole_audio=True, use_aug=False, rank=rank)

    data_loader = DataLoader(train_dataset, batch_size, drop_last=True,
                             pin_memory=True, persistent_workers=True, num_workers=4,
                             sampler=DistributedSampler(train_dataset))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    model = DDP(E2E0(4, 1, (2, 2)).to(device),find_unused_parameters=True)
    if resume_iteration is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model_{resume_iteration}.pt')
        ckpt = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(ckpt['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    if not rank:
        summary(model.module)
        loop = tqdm(range(resume_iteration + 1, iterations + 1))
    else:
        loop = range(resume_iteration + 1, iterations + 1)
    RPA, RCA, OA, VFA, VR, it = 0, 0, 0, 0, 0, 0

    for i, data in zip(loop, cycle(data_loader)):
        mel = data['mel'].to(device)
        pitch_label = data['pitch'].to(device)
        pitch_pred = model(mel)
        loss = bce(pitch_pred, pitch_label)


        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        scheduler.step()
        writer.add_scalar('loss/loss_pitch', loss.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                metrics = evaluate(validation_dataset, model, hop_length, device)
                for key, value in metrics.items():
                    writer.add_scalar('stage_pitch/' + key, np.mean(value), global_step=i)
                rpa = np.mean(metrics['RPA'])
                rca = np.mean(metrics['RCA'])
                oa = np.mean(metrics['OA'])
                vr = np.mean(metrics['VR'])
                vfa = np.mean(metrics['VFA'])
                RPA, RCA, OA, VR, VFA, it = rpa, rca, oa, vr, vfa, i
                with open(os.path.join(logdir, 'result.txt'), 'a') as f:
                    f.write(str(i) + '\t')
                    f.write(str(RPA) + '\t')
                    f.write(str(RCA) + '\t')
                    f.write(str(OA) + '\t')
                    f.write(str(VR) + '\t')
                    f.write(str(VFA) + '\n')
                torch.save({'model': model.state_dict()}, os.path.join(logdir, f'model_{i}.pt'))
            model.train()
        logger.log(
            step = i,
            data = {
                    "loss":loss.item(), 
                    },
            verbosity=Verbosity.DEFAULT,
        )
        if not rank:
            loop.set_postfix(迭代次数=i,总损失=loss.item())

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        required=False,
        default='runs/Hybrid_bce',
        help="path to log dir",
    )
    parser.add_argument(
        "-hop",
        "--hop_length",
        type=int,
        required=False,
        default=160,
        help="hop_length under 16khz sampling rate | default: 160",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        required=False,
        default=5e-4,
        help="learning rate while training",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        required=False,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "-vi",
        "--validation_interval",
        type=int,
        required=False,
        default=50,
        help="number of validation intervals",
    )
    parser.add_argument(
        "-cgn",
        "--clip_grad_norm",
        type=int,
        required=False,
        default=3,
        help="max norm of the gradients",
    )
    parser.add_argument(
        "-npn",
        "--nproc_per_node",
        type=int,
        required=False,
        default=1,
        help="world size in DDP",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        required=False,
        default=100,
        help="total train steps, default train 100 steps",
    )
    parser.add_argument(
        "-lrds",
        "--learning_rate_decay_steps",
        type=int,
        required=False,
        default=2000,
        help="learning rate decay steps",
    )
    parser.add_argument(
        "-lrdr",
        "--learning_rate_decay_rate",
        type=float,
        required=False,
        default=0.98,
        help="total train steps, default train 100 steps",
    )
    parser.add_argument(
        "-ri",
        "--resume_iteration",
        type=int,
        required=False,
        default=None,
        help="appoint an iteration step for continue training, default to None, which means training from scratch",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        required=False,
        default="E2E0",
        help="appoint an iteration step for continue training, default to None, which means training from scratch",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=torch.device('sdaa' if torch.sdaa.is_available() else 'cpu'),
        required=False,
        help="cpu or sdaa, auto if not set")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1145141919810,
        required=False,
        help="random seed for PyTorch")
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    cmd = parse_args()
    logdir = cmd.logdir
    hop_length = cmd.hop_length
    learning_rate = cmd.learning_rate
    batch_size = cmd.batch_size
    validation_interval = cmd.validation_interval
    clip_grad_norm = cmd.clip_grad_norm
    world_size = cmd.nproc_per_node
    iterations = cmd.iterations
    learning_rate_decay_steps = cmd.learning_rate_decay_steps
    learning_rate_decay_rate = cmd.learning_rate_decay_rate
    resume_iteration = cmd.resume_iteration
    model_name = cmd.model_name
    torch.manual_seed(cmd.seed)
    if torch.sdaa.is_available():
        torch.sdaa.manual_seed(cmd.seed)
    
    assert model_name == "E2E0"
    device = torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')

    args = (logdir, hop_length, learning_rate,
          batch_size, validation_interval, clip_grad_norm,
          world_size, iterations, learning_rate_decay_steps,
          learning_rate_decay_rate, resume_iteration, device)
    mp.spawn(train,
             args=args,
             nprocs=world_size,
             join=True)
