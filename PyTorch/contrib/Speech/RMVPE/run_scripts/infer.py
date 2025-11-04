import importlib
import os
import warnings
import sys
from pathlib import Path


root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))


import numpy as np
import librosa
import time
import argparse
import torch
from src import RMVPE

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output csv file",
    )
    parser.add_argument(
        "-hop",
        "--hop_length",
        type=str,
        required=False,
        default=160,
        help="hop_length under 16khz sampling rate | default: 160",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=str,
        required=False,
        default=0.03,
        help="unvoice threhold | default: 0.03",
    )
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    cmd = parse_args()
    
    model_path = cmd.model
    device = cmd.device
    audio_path = cmd.input
    output_path = cmd.output
    hop_length = int(cmd.hop_length)
    thred = float(cmd.threhold)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('loading model and audio')
    rmvpe = RMVPE(model_path, hop_length=hop_length)
    audio, sr = librosa.load(audio_path, sr=None)
    print('start infering ...')
    t = time.time()
    f0 = rmvpe.infer_from_audio(audio, sr, device=device, thred=thred, use_viterbi=False)
    infer_time = time.time() - t
    print('time: ', infer_time)
    print('RTF: ', infer_time * sr / len(audio))
    np.savetxt(output_path, np.array([0.01 * np.arange(len(f0)), f0]).transpose(),delimiter=',')
