import torch
import argparse
from argparse import ArgumentParser,ArgumentTypeError
import sys
import warnings


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc_per_node',type=int,default=1)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--use_amp", required=False, default=False, type=str2bool, help='Distributed training or not')
    parser.add_argument("--use_ddp", required=False, default=False, type=str2bool, help='DDP training or not')
    parser.add_argument("--local-rank", required=False, default=-1, type=int)
    return parser.parse_args()


def check_argument(args):
    if args.step>0 and args.epoch>1:
        args.epoch=1
        warnings.warn('argument step and epoch is conflict, when step is used, epoch will be set to 1')
    return args


if __name__ == '__main__':
    sys.exit(0)
