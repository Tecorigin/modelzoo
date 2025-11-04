import csv
import math
import os
import random
import paddle.audio as paddleaudio
from librosa import resample
import numpy as np
import soundfile as sf
import paddle
from paddle.io import Dataset
from tqdm import tqdm
from scipy import signal
import paddle.nn.functional as Func
from numpy import random
import ffmpeg
from scipy.signal.windows import hann
from util import stft, istft

data_buffer = dict()

def _load_audio(
        path, offset=None, duration=None,
        sample_rate=None, dtype=np.float32, use_paddle_direct=True):
    """ Loads the audio file denoted by the given path
    and returns it data as a waveform.

    :param path: Path of the audio file to load data from.
    :param offset: (Optional) Start offset to load from in seconds.
    :param duration: (Optional) Duration to load in seconds.
    :param sample_rate: (Optional) Sample rate to load audio with.
    :param dtype: (Optional) Numpy data type to use, default to float32.
    :returns: Loaded data a (waveform, sample_rate) tuple.
    :raise SpleeterError: If any error occurs while loading audio.
    """
    global data_buffer
    if use_paddle_direct:
        if not((data:=data_buffer.get(path)) is None): # 如果读取过数据
            data, sr = data_buffer[path]
            offset = int(offset * sr)
            duration = int(duration * sr)
            return data[::,offset:offset + duration].T, sr
        else: # 如果没有读取过数据
            data, sr = paddle.audio.load(path) # 读取数据
            del data_buffer# 先清除缓存
            data_buffer = dict()
            data_buffer[path] = (data, sr) # 然后存缓存
            offset = int(offset * sr)
            duration = int(duration * sr)
            return data[::,offset:offset + duration].T, sr
    
    if not isinstance(path, str):
        path = path.decode()
        
    probe = ffmpeg.probe(path)

    metadata = next(
        stream
        for stream in probe['streams']
        if stream['codec_type'] == 'audio')
    n_channels = metadata['channels']
    if sample_rate is None:
        sample_rate = metadata['sample_rate']
    output_kwargs = {'format': 'f32le', 'ar': sample_rate}
    if duration is not None:
        output_kwargs['t'] = _to_ffmpeg_time(duration)
    if offset is not None:
        output_kwargs['ss'] = _to_ffmpeg_time(offset)
    process = (
        ffmpeg
        .input(path)
        .output('pipe:', **output_kwargs)
        .run_async(pipe_stdout=True, pipe_stderr=True))
    buffer, _ = process.communicate()
    waveform = np.frombuffer(buffer, dtype='<f4').reshape(-1, n_channels)
    if not waveform.dtype == np.dtype(dtype):
        waveform = waveform.astype(dtype)

    return paddle.to_tensor(waveform,dtype = paddle.float32), sample_rate
 
def _to_ffmpeg_time(n):
    """ Format number of seconds to time expected by FFMPEG.
    :param n: Time in seconds to format.
    :returns: Formatted time in FFMPEG format.
    """
    m, s = divmod(n, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%09.6f' % (h, m, s)




def _stft(data, inverse=False, frame_length=4096, frame_step=1024, length=None):
    """
    Single entrypoint for both stft and istft. This computes stft and istft with librosa on stereo data. The two
    channels are processed separately and are concatenated together in the result. The expected input formats are:
    (n_samples, 2) for stft and (T, F, 2) for istft.
    :param data: np.array with either the waveform or the complex spectrogram depending on the parameter inverse
    :param inverse: should a stft or an istft be computed.
    :return: Stereo data as numpy array for the transform. The channels are stored in the last dimension
    """
    assert not (inverse and length is None)
    N = frame_length
    H = frame_step
    win = paddleaudio.functional.get_window('hann',N,False)
    fstft = istft if inverse else stft
    win_len_arg = {"win_length": None, "length": length,"n_fft": N} if inverse else {"n_fft": N}
    n_channels = data.shape[-1]
    out = []
    for c in range(n_channels):
        d = data[:, :, c].T if inverse else data[:, c]
        s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
        s = paddle.unsqueeze(s.T, 2-inverse)
        out.append(s)
    del data
    if len(out) == 1:
        return out[0]
    return paddle.concat(out, axis=2-inverse)


class TrainDataset(Dataset):
    def __init__(self, params):
        self.datasets = []
        self.count = 0
        self.MARGIN = params.margin
        self.chunk_duration = params.chunk_duration
        self.n_chunks_per_song = params.n_chunks_per_song
        self.frame_length = params.frame_length
        self.frame_step = params.frame_step
        self.T = params.T
        self.F = params.F

        with open(params.train_manifest, 'r') as f:
            reader = csv.reader(f)
            for mix_path, vocal_path, instrumental_path, duration, samplerate in reader:
                duration = float(duration)
                for k in range(self.n_chunks_per_song):
                    if self.n_chunks_per_song > 1:
                        start_time = k * (duration - self.chunk_duration - 2 * self.MARGIN) / (self.n_chunks_per_song - 1) + self.MARGIN
                        if start_time > 0.0:
                            self.datasets.append((mix_path, vocal_path, instrumental_path, duration, samplerate, start_time))
                            self.count += 1
                    elif self.n_chunks_per_song == 1:
                        start_time = duration / 2 - self.chunk_duration / 2
                        if start_time > 0.0:
                            self.datasets.append((mix_path, vocal_path, instrumental_path, duration, samplerate, start_time))
                            self.count += 1
        
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, chunk_id):
        chunk_id %= self.count
        pair = self.datasets[chunk_id]
        mix_chunk = pair[0]
        vocal_chunk = pair[1]
        instru_chunk = pair[2]
        samplerate = float(pair[4])
        start_time = float(pair[5])
                
        ### load audio ### 
        mix_audio, mix_sr = _load_audio(mix_chunk, offset=start_time, duration=self.chunk_duration) 
        vocal_audio, vocal_sr = _load_audio(vocal_chunk, offset=start_time, duration=self.chunk_duration)
        instru_audio, instru_sr = _load_audio(instru_chunk, offset=start_time, duration=self.chunk_duration)

        mix_audio = mix_audio.T
        vocal_audio = vocal_audio.T
        instru_audio = instru_audio.T

        ### 2 channels ###
        if mix_audio.shape[0] == 1: 
            mix_audio = paddle.concat((mix_audio, mix_audio), axis=0)
        if vocal_audio.shape[0] == 1:
            vocal_audio = paddle.concat((vocal_audio, vocal_audio), axis=0)
        if instru_audio.shape[0] == 1:
            instru_audio = paddle.concat((instru_audio, instru_audio), axis=0)
        if mix_audio.shape[0] > 2:
            mix_audio = mix_audio[:2, :]
        if vocal_audio.shape[0] > 2:
            vocal_audio = vocal_audio[:2, :]
        if instru_audio.shape[0] > 2:
            instru_audio = instru_audio[:2, :]

        ### resample ###
        if int(samplerate) != 44100:
            mix_audio = paddle.to_tensor(resample(mix_audio.numpy(),int(samplerate), 44100))
            vocal_audio = paddle.to_tensor(resample(vocal_audio.numpy(),int(samplerate), 44100))
            instru_audio = paddle.to_tensor(resample(instru_audio.numpy(),int(samplerate), 44100))
            samplerate = 44100
    
        
        ### stft ###
        mix_stft = _stft(mix_audio.T, frame_length=self.frame_length, frame_step=self.frame_step)
        mix_stft_mag = paddle.abs(mix_stft)
        mix_stft_mag = mix_stft_mag.transpose([2, 1, 0])
        
        vocal_stft = _stft(vocal_audio.T, frame_length=self.frame_length, frame_step=self.frame_step)
        vocal_stft_mag = paddle.abs(vocal_stft)
        vocal_stft_mag = vocal_stft_mag.transpose([2, 1, 0])

        instru_stft = _stft(instru_audio.T, frame_length=self.frame_length, frame_step=self.frame_step)
        instru_stft_mag = paddle.abs(instru_stft)
        instru_stft_mag = instru_stft_mag.transpose([2, 1, 0])

        num_frame = mix_stft_mag.shape[2]
        start = random.randint(low=1, high=(num_frame - self.T))
        end = start + self.T
        mix_stft_mag = mix_stft_mag[:, :self.F, start: end]
        vocal_stft_mag = vocal_stft_mag[:, :self.F, start: end]
        instru_stft_mag = instru_stft_mag[:, :self.F, start: end]
                
        return mix_stft_mag, vocal_stft_mag, instru_stft_mag
                


