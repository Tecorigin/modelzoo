import importlib
import os
import warnings
import sys
from pathlib import Path


root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

import argparse
import os
import sys

import numpy as np
import soundfile
import librosa
import paddle

paddle.set_device("cpu")
from mir_eval.util import midi_to_hz

from onsets_and_frames import *

model_complexity = 48
def load_and_process_audio(flac_path, sequence_length, device):

    random = np.random.RandomState(seed=42)

    audio, sr = soundfile.read(flac_path, dtype='float32')
    if len(audio.shape) == 2:
        audio = audio.mean(1)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = (audio * 32768.0).astype('int16')
    sr = SAMPLE_RATE
    assert sr == SAMPLE_RATE,(sr, SAMPLE_RATE)

    audio = paddle.to_tensor(audio,dtype = paddle.int16)

    if sequence_length is not None:
        audio_length = len(audio)
        step_begin = random.randint(audio_length - sequence_length) // HOP_LENGTH
        n_steps = sequence_length // HOP_LENGTH

        begin = step_begin * HOP_LENGTH
        end = begin + sequence_length

        audio = audio[begin:end]#.to(device)
    else:
        # audio = audio.to(device)
        pass

    audio = audio.astype(paddle.float32).divide(paddle.to_tensor(32768.0))

    return audio


def transcribe(model, audio):

    # model = model.to('cpu')
    #melspectrogram = melspectrogram.to('cpu')
    # melspectrogram.to('cpu')
    mel = melspectrogram(audio.reshape([-1, audio.shape[-1]])[:, :-1]).transpose([0,2,1])
    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

    predictions = {
            'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
            'offset': offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2])),
            'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
            'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
        }

    return predictions


def transcribe_file(model_file, flac_paths, save_path, sequence_length,
                  onset_threshold, frame_threshold, device):
    paddle.set_device(device)
    model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity)
    model.set_state_dict(paddle.load(model_file))
    model.eval()
    # model.to(device) # 会有离谱的报错
    #summary(model)

    for flac_path in flac_paths:
        print(f'处理 {flac_path} 中...', file=sys.stderr)
        audio = load_and_process_audio(flac_path, sequence_length, device)
        predictions = transcribe(model, audio)

        p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'], onset_threshold, frame_threshold)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_est = (i_est * scaling).reshape([-1, 2])
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        os.makedirs(save_path, exist_ok=True)
        pred_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.png')
        save_pianoroll(pred_path, predictions['onset'], predictions['frame'])
        midi_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('flac_paths', type=str, nargs='+')
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cpu')

    with paddle.no_grad():
        transcribe_file(**vars(parser.parse_args()))
