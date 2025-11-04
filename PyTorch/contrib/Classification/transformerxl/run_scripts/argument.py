# argument.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

    # 数据相关参数
    parser.add_argument('--data', type=str, default='/data/teco-data/enwik8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='enwik8',
                        choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')

    # 模型相关参数
    parser.add_argument('--n_layer', type=int, default=12,
                        help='number of total layers')
    parser.add_argument('--n_head', type=int, default=10,
                        help='number of heads')
    parser.add_argument('--d_model', type=int, default=500,
                        help='model dimension')
    parser.add_argument('--d_head', type=int, default=50,
                        help='head dimension')
    parser.add_argument('--d_inner', type=int, default=1000,
                        help='inner dimension in FF')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='global dropout rate')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='attention probability dropout rate')

    # 训练相关参数
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    parser.add_argument('--optim', default='adam', type=str,
                        choices=['adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='upper epoch limit')
    parser.add_argument('--max_step', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--tgt_len', type=int, default=70,
                        help='number of tokens to predict')
    parser.add_argument('--eval_tgt_len', type=int, default=50,
                        help='number of tokens to predict for evaluation')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--mem_len', type=int, default=0,
                        help='length of the retained previous heads')

    # 其他参数
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--fp16', action='store_true',
                        help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')


    args = parser.parse_args()
    return args