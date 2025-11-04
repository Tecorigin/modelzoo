import torch
import argparse
import sys
import os
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Transfer to CIFAR Training')
    parser.add_argument('--seed', type=int, default=12, help='random seed')
    parser.add_argument('--master_port', type=int, default=29501, help='master port')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--grad_scale', type=str, help='grad scale')
    parser.add_argument('--autocast', type=str, help='autocast')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, or cinic10')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--n_gpus', type=int, default=1, help='number of available gpus for training')
    parser.add_argument('--lr', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--drop', type=float, default=0.2, help='drop out rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--autoaugment', action='store_true', default=False, help='use auto augmentation')
    parser.add_argument('--save', action='store_true', default="./pt", help='dump output')
    parser.add_argument('--model', type=str, default=None,
                    help='location of a json file of specific model declaration')
    parser.add_argument('--imagenet', type=str, default=None,
                    help='location of initial weight to load')
                    
    parser.add_argument("-distributed", action='store_true', help="Whether to run training.")

    parser.add_argument('--nproc_per_node', default=3, type=int,
                    help="The number of processes to launch on each node, "
                    "for GPU training, this is recommended to be set "
                    "to the number of GPUs in your system so that "
                    "each process can be bound to a single GPU.")
    return parser.parse_args()
  
  
if __name__ == '__main__':
    args = parse_args()
    
    
    model_name = args.model_name
    bs = args.batch_size
    epochs = args.epochs
    nnode = 1
    nproc_per_node = args.nproc_per_node
    lr = args.lr
    master_port = 29501
    autocast = args.autocast
    dataset = args.dataset
    current_file_directory = Path(__file__).resolve().parent
    if nnode > 1:
        raise Exception("Recent task do not support nnode > 1. Set --nnode=1 !")

    if 'mobilenet' not in model_name:
        raise ValueError('please use mobilenet model')


    if nnode == 1 and nproc_per_node > 1:
        cmd = f'torchrun --nproc_per_node {nproc_per_node} --master_port {master_port} {current_file_directory}/../trainWithDDP.py \
              --model_name {model_name} \
              -distributed \
              --epochs {epochs} \
              --dataset {dataset} \
              -nproc_per_node {nproc_per_node} \
              --batch_size {bs} \
              --lr {lr} '
        if autocast:
            cmd += ' --autocast True'

    else:
        cmd = f'python {current_file_directory}/../trainWithDDP.py \
              --model_name {model_name} \
              --epochs {epochs} \
              --dataset {dataset} \
              --batch_size {bs} \
              --lr {lr} '
        if autocast:
            cmd += ' --autocast True'


    os.system(cmd)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    