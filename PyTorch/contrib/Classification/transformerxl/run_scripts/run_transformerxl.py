import os
import sys
from argument import parse_args
import subprocess

def build_cmd(args):
    cmd = [
        "python", "../pytorch/train.py",
        '--data', args.data,
        '--dataset', args.dataset,
        '--n_layer', str(args.n_layer),
        '--n_head', str(args.n_head),
        '--d_model', str(args.d_model),
        '--d_head', str(args.d_head),
        '--d_inner', str(args.d_inner),
        '--dropout', str(args.dropout),
        '--dropatt', str(args.dropatt),
        '--lr', str(args.lr),
        '--optim', args.optim,
        '--scheduler', args.scheduler,
        '--warmup_step', str(args.warmup_step),
        '--max_step', str(args.max_step),
        '--batch_size', str(args.batch_size),
        '--tgt_len', str(args.tgt_len),
        '--eval_tgt_len', str(args.eval_tgt_len),
        '--ext_len', str(args.ext_len),
        '--mem_len', str(args.mem_len),
        '--seed', str(args.seed),

    ]

    if args.debug:
        cmd.append('--debug')
    if args.fp16:
        cmd.append('--fp16')

    return cmd

def main():
    args = parse_args()
    cmd = build_cmd(args)

    print(" Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
