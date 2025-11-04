from datetime import datetime
import argparse
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--logdir', type=str, default=(logdir := 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')))
    parser.add_argument('-d','--device', type=str, default=(device := 'sdaa'))#if paddle.device.is_compiled_with_cuda() else 'cpu'
    parser.add_argument('-npn','--nproc_per_node', type=int, default=(nproc_per_node := 1))
    parser.add_argument('-i','--iterations', type=int, default=(iterations := 500000))
    parser.add_argument('-ri','--resume_iteration', type=int, default=(resume_iteration := None))
    parser.add_argument('-ci','--checkpoint_interval', type=int, default=(checkpoint_interval := 1000))
    parser.add_argument('-to','--train_on', type=str, default=(train_on := 'MAPS')) # or 'MAESTRO'
    parser.add_argument('-bs','--batch_size', type=int, default=(batch_size := 8))
    parser.add_argument('-sl','--sequence_length', type=int, default=(sequence_length := 327680))
    parser.add_argument('-mc','--model_complexity', type=int, default=(model_complexity := 48)) # 这个就不要改了
    parser.add_argument('-a','--amp_on', type=bool, default=(amp_on := False))
    parser.add_argument('-lr','--learning_rate', type=float, default=(learning_rate := 6e-4))
    parser.add_argument('-lrds','--learning_rate_decay_steps', type=float, default=(learning_rate_decay_steps := 10000))
    parser.add_argument('-lrdr','--learning_rate_decay_rate', type=float, default=(learning_rate_decay_rate := 0.98))
    parser.add_argument('-loo','--leave_one_out', type=int, default=(leave_one_out := None))
    parser.add_argument('-cgn','--clip_gradient_norm', type=int, default=(clip_gradient_norm := 3))
    parser.add_argument('-vl','--validation_length', type=int, default=(validation_length := sequence_length))
    parser.add_argument('-vi','--validation_interval', type=int, default=(validation_interval := iterations+10)) #500 original, avoid test on sdaa

    return parser.parse_args()