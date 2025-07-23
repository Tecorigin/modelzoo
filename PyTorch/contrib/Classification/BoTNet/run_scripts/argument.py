import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--img_size', default=224, type=int,
                        help='Image size to which input images should be scaled.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch Size for training.')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='Learning rate for training model.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Epoch number for training model.')
    parser.add_argument('--output', default="../", type=str,
                        help='Where to output all stuff')
    parser.add_argument('--num_heads', default=4, type=int,
                        help='Number of heads in attention layers.')
    parser.add_argument('--name', default="", type=str,
                        help='Add a name ending to saved model and log file.')
    parser.add_argument('--num_steps', default=100, type=int,
                        help='train iteration limits')
    parser.add_argument('--time', default="", type=str,
                        help='Timestamp for model and log file naming.')
    return parser

if __name__ == "__main__":
    import sys
    sys.exit(0)