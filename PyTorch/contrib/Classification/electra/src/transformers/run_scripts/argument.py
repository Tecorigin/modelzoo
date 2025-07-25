# argument.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='google/electra-small-discriminator')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    return parser.parse_args()
