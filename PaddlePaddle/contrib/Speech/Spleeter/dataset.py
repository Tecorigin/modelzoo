import csv
import math
import os
import random
import paddle.audio as paddleaudio
import numpy as np
import paddle
import util
from paddle.io import Dataset
from librosa import resample
import paddle.nn.functional as Func
from numpy import random



def load_audio(filename, start = 0, stop = None, resample = True):
    '''
        Load wav file
        Args:
            filename  :  音频路径
            start     :  开始帧(integer or None)
            stop      :  结束帧(integer or None)
        Return:
            y         :  双通道 (L x 2)
    '''
    #y = None
    
    #y, _ = sf.read(filename, start = start, stop = stop, dtype = 'float32', always_2d = True)
    #y = y[:, : 2]

    wav, sr = paddleaudio.load_wav(filename)
    
    wav_torch = wav / (wav.max() + 1e-8)
    if start != None:
        wav_torch = wav_torch[:, start: stop]
    
    return wav_torch, sr

def stft_feature(
        waveform, sample_rate=44100, frame_length=2048, frame_step=512,
        spec_exponent=1., F=1024, T=512, separate=False):
    '''
        Computes stft feature from wav
        Args:
            waveform      :    双通道 (L x 2)
            frame_length  :    帧长
            frame_step    :    帧移
            spec_exponent :    频谱指数(1为幅值谱 2为能量谱)
            F             :    输出时频矩阵的频域点数
            T             :    输出时频矩阵的时域点数
        Return:
                输出频谱(F x T x 2).
    '''


    stft = util.stft(
            waveform, frame_length, hop_length=frame_step, window=paddle.audio.functional.get_window('hann', frame_length))

        # only keep freqs smaller than self.F
    
    stft = stft[:, :F, : , :]
    real = stft[:, :, :, 0]
    im = stft[:, :, :, 1]
    mag = paddle.sqrt(real ** 2 + im ** 2)

    return stft, mag

def pad_and_partition(tensor, T):
    """
    pads zero and partition tensor into segments of length T
    Args:
        tensor(Tensor): BxCxFxL
    Returns:
        tensor of size (B*[L/T] x C x F x T)
    """
    
    old_size = tensor.size(3)
    new_size = math.ceil(old_size/T) * T
    tensor = Func.pad(tensor, [0, new_size - old_size])
    [b, c, t, f] = tensor.shape
    print('dataset.py 79 split == tensor.shape[3]//T：',split == tensor.shape[3]//T)
    split = new_size // T
    
    return paddle.concat(paddle.split(tensor, tensor.shape[3]//T, axis=3), axis=0)


class TrainDataset(Dataset):
    def __init__(self, params):
        self.datasets = []
        self.count = 0
        self.MARGIN = params['margin']
        self.chunk_duration = params['chunk_duration']
        self.n_chunks_per_song = params['n_chunks_per_song']
        self.frame_length = params['frame_length']
        self.frame_step = params['frame_step']
        self.T = params['T']
        self.F = params['F']

        with open(params['train_manifest'], 'r') as f:
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
                
         
        ### resample ###
        if int(samplerate) != 44100:
            mix_audio = resample(mix_audio,int(samplerate), 44100)
            vocal_audio = resample(vocal_audio, int(samplerate), 44100)
            instru_audio = resample(instru_audio, int(samplerate), 44100)
            samplerate = 44100
        
        mix_audio, mix_sr = load_audio(mix_chunk, start=int(start_time * samplerate), stop=int((start_time + self.chunk_duration) * samplerate))
        vocal_audio, vocal_sr = load_audio(vocal_chunk, start=int(start_time * samplerate), stop=int((start_time + self.chunk_duration) * samplerate))
        instru_audio, instru_sr = load_audio(instru_chunk, start=int(start_time * samplerate), stop=int((start_time + self.chunk_duration) * samplerate))
       
        ### 2 channels ###
        if mix_audio.shape[0] == 1: 
            mix_audio = paddle.concat((mix_audio, mix_audio), dim=0)
            vocal_audio = paddle.concat((vocal_audio, vocal_audio), dim=0)
            instru_audio = paddle.concat((instru_audio, instru_audio), dim=0)
        elif mix_audio.shape[0] > 2:
            mix_audio = mix_audio[:2, :]
            vocal_audio = vocal_audio[:2, :]
            instru_audio = instru_audio[:2, :]

        ### stft ###
        mix_stft, mix_stft_mag = stft_feature(mix_audio, sample_rate=samplerate, frame_length=self.frame_length, frame_step=self.frame_step, spec_exponent=1., F=self.F, T=self.T)
        vocal_stft, vocal_stft_mag = stft_feature(vocal_audio, sample_rate=samplerate, frame_length=self.frame_length, frame_step=self.frame_step, spec_exponent=1., F=self.F, T=self.T)
        instru_stft, instru_stft_mag = stft_feature(instru_audio, sample_rate=samplerate, frame_length=self.frame_length, frame_step=self.frame_step, spec_exponent=1., F=self.F, T=self.T)
        
        ### random_time_crop ###
        num_frame = mix_stft_mag.shape[2]
        start = random.randint(low=1, high=(num_frame - self.T))
        end = start + self.T
        mix_stft_mag = mix_stft_mag[:, :, start: end]
        vocal_stft_mag = vocal_stft_mag[:, :, start: end]
        instru_stft_mag = instru_stft_mag[:, :, start: end]

        return mix_stft_mag, vocal_stft_mag, instru_stft_mag
                


class SeparateDataset(Dataset):
    def __init__(self, params):
        self.datasets = []
        self.count = 0
        self.chunk_duration = params['chunk_duration']
        self.n_chunks_per_song = params['n_chunks_per_song']
        self.frame_length = params['frame_length']
        self.frame_step = params['frame_step']
        self.T = params['T']
        self.F = params['F']

        with open(params['separate_manifest'], 'r') as f:
            reader = csv.reader(f)
            for path, duration, samplerate in reader:
                self.datasets.append((path, duration, samplerate))
                self.count += 1
        
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, audio_id):
        audio_id %= self.count
        audio = self.datasets[audio_id]

        path = audio[0]
        duration = float(audio[1])
        samplerate = float(audio[2])
        wav_name = path.split('/')[-1].split('.')[0]

        source_audio, _ = load_audio(path)
        stft, stft_mag = stft_feature(source_audio, sample_rate=samplerate, frame_length=self.frame_length, frame_step=self.frame_step, spec_exponent=1., F=self.F, T=self.T) # 2 * F * L
        stft_mag = stft_mag.unsqueeze(-1).permute([3, 0, 1, 2])
            
        L = stft.size(2)

        stft_mag = pad_and_partition(stft_mag, self.T) # [(L + T) / T] * 2 * F * T

        return stft, stft_mag.transpose(2, 3), L, wav_name, samplerate
