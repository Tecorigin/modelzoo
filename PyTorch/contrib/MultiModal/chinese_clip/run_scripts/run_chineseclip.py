from argument import parse_args, check_argument
import os
from pathlib import Path

if __name__ == '__main__':
    args = parse_args()
    args = check_argument(args)

    model_name = args.model_name
    bs = args.batch_size
    epoch = args.epoch
    nnode = 1
    nproc_per_node = args.nproc_per_node
    node_rank = args.node_rank
    lr = args.lr
    master_port = 29501
    ckpt_path = args.checkpoint_path
    bert_ckpt_path = args.bert_ckpt_path
    do_predict = args.do_predict
    autocast = args.autocast
    dataset_path = args.datasets_path
    device = args.device

    current_file_directory = Path(__file__).resolve().parent

    output_dir =  os.path.join(current_file_directory, '../logs/chinese_clip_multi')
    output_dir_single = os.path.join(current_file_directory, '../logs/chinese_clip_single')
    print(output_dir)
    if nnode > 1:
        raise Exception("Recent task do not support nnode > 1. Set --nnode=1 !")

    if 'chinese_clip' not in model_name:
        raise ValueError('please use chinese_clip model')

    if nnode == 1 and nproc_per_node > 1:
        cmd = f'torchrun --nproc_per_node {nproc_per_node} --master_port {master_port} {current_file_directory}/../main.py \
              --model_name {model_name} \
              --output_dir {output_dir} \
              --do_train \
              --distributed \
              --ckpt_path {ckpt_path} \
              --bert_ckpt_path {bert_ckpt_path} \
              --device {device} \
              --epoch {epoch} \
              --datasets_path {dataset_path} \
              --nproc_per_node {nproc_per_node} \
              --batch_size {bs} \
              --learning_rate {lr} '
        if autocast:
            cmd += ' --amp'
        if do_predict:
            cmd += '--do_predict'

    else:
        cmd = f'python {current_file_directory}/../main.py \
              --model_name {model_name} \
              --output_dir {output_dir_single} \
              --do_train \
              --ckpt_path {ckpt_path} \
              --bert_ckpt_path {bert_ckpt_path} \
              --device {device} \
              --epoch {epoch} \
              --datasets_path {dataset_path} \
              --batch_size {bs} \
              --learning_rate {lr} '
        if autocast:
            cmd += ' --amp'
        if do_predict:
            cmd += '--do_predict'

    os.system(cmd)
