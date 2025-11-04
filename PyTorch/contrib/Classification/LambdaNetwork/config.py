import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--print_intervals', type=int, default=100)
    parser.add_argument('--evaluation', type=bool, default=False)
    parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--gradient_clip', type=float, default=2)
    parser.add_argument('--num_steps', type=int, default=300, help='Number of steps for training')

    return parser.parse_args()
