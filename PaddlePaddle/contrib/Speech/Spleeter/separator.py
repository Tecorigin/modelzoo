import paddle
import os
import paddle.nn.functional as Func
import numpy as np
import paddle.nn as nn
from model import UNet
import soundfile as sf
import paddle.audio as paddleaudio
import librosa

from util import stft
from util import istft

class Separator(nn.Layer):
    def __init__(self, T = 512, F = 1024, frame_length = 4096, frame_step = 1024, num_instruments = 2, resume = ['./model/net_vocal_0.pdparams', './model/net_instrumental_0.pdparams']):
        super().__init__()
        self.num_instruments = num_instruments
        self.model_list = nn.LayerList()
        for i in range(self.num_instruments):
            checkpoint = paddle.load(resume[i])['state_dict'] if isinstance(resume[i],str) else resume[i]
            net = UNet()
            net.set_state_dict(checkpoint)
            #net.to(device)
            self.model_list.append(net)

        self.T = T
        self.F = F
        self.frame_length = frame_length
        self.frame_step = frame_step



    def _stft(self, data, inverse=False, length=None):
        """
        Single entrypoint for both stft and istft. This computes stft and istft with librosa on stereo data. The two
        channels are processed separately and are concatenated together in the result. The expected input formats are:
        (n_samples, 2) for stft and (T, F, 2) for istft.
        :param data: np.array with either the waveform or the complex spectrogram depending on the parameter inverse
        :param inverse: should a stft or an istft be computed.
        :return: Stereo data as numpy array for the transform. The channels are stored in the last dimension
        """
        assert not (inverse and length is None)
        
        N = self.frame_length
        H = self.frame_step
        win = paddleaudio.functional.get_window('hann',N,False) if not inverse else None
        win_len_arg = {"length": length,"n_fft": N,} if inverse else {"n_fft": N,'window':win.astype(paddle.float32),}#
        fstft = istft if inverse else stft
        n_channels = data.shape[-1]
        out = []
        for c in range(n_channels):
            d = data[:, :, c].T if inverse else data[:, c]
            s = fstft(d, hop_length=H, center=False, **win_len_arg)
            s = paddle.unsqueeze(s.T, 2-inverse)
            if s.dtype != paddle.float32:
                out.append(paddle.as_real(s))
            else:
                out.append((s))
        if len(out) == 1:
            return paddle.as_complex(out[0]) if out[0].dtype != paddle.float32 else out[0]
        ret = paddle.concat(out, axis=2-inverse)
        ret = paddle.as_complex(ret) if not inverse else ret
        return ret

    def _pad_and_partition(self, tensor:paddle.Tensor, T):
        old_size = tensor.shape[3]
        new_size = float(paddle.ceil(paddle.to_tensor(old_size/T)) * T)
        temp = paddle.to_tensor([0, new_size - old_size,0,0], dtype=paddle.int32)
        tensor = Func.pad(tensor, temp)
        [b, c, t, f] = tensor.shape
        split = paddle.to_tensor(new_size / T).astype(paddle.int32)
        if int(split) == 1:
            return tensor
        split_value = paddle.to_tensor(tensor.shape[3]//int(split))
        split_shape = paddle.full([split],split_value,'int32')
        # assert int(split) == 1,'导出静态图时没有办法解决的问题'
        return paddle.concat(paddle.split(tensor, list(split_shape.numpy()), axis=3), axis=0)

    def forward(self, source_audio, samplerate=44100):

        assert int(samplerate) == 44100,'sample rate must be 44100'

        if source_audio.shape[0] == 1: 
            source_audio = paddle.concat((source_audio, source_audio), axis=0)
        elif source_audio.shape[0] > 2:
            source_audio = source_audio[:2, :]
        stft = self._stft(source_audio.T) # L * F * 2
        stft = stft[:, : self.F, :]
        stft_mag = paddle.abs(stft) # L * F * 2 
        # 静态图多一个维度，不知道为什么
        if len(stft_mag.shape) == 4:
            stft_mag = stft_mag.squeeze(-1)
            stft_mag = stft_mag.unsqueeze(0)

        if len(stft_mag.shape) == 3:
            stft_mag = stft_mag.unsqueeze(0)
        stft_mag = stft_mag.transpose([0, 3, 2, 1]) # 1 * 2 * F * L
        L = stft.shape[0]
        stft_mag = self._pad_and_partition(stft_mag, self.T) # [(L + T) / T] * 2 * F * T
        stft_mag = stft_mag.transpose([0,1,3,2])
        # stft_mag : B * 2 * T * F
        B = stft_mag.shape[0]
        masks = []        

        for model in self.model_list:  
            mask = model(stft_mag, output_mask_logit=True) 
            masks.append(mask) 

        mask_sum = sum([m ** 2 for m in masks]) 
        mask_sum += 1e-10 
        return_list = []
        for i in range(self.num_instruments):
            mask = masks[i]
            fir = (mask ** 2 + 1e-10/2)
            mask = fir / (mask_sum)
            mask = mask.transpose([0,1,3,2])  # B x 2 X F x T
            if mask.shape[0] == 1:
                mask = mask
            else:
                mask = paddle.concat(paddle.split(mask, mask.shape[0], axis=0), axis=3)
            mask = mask.squeeze(0)[:,:,:L] # 2 x F x L
            mask = mask.transpose([2, 1, 0])

            if len(stft.shape) == 4 and stft.shape[-1] == 1:
                stft = stft.squeeze(3)
            stft_masked = (stft.as_real() * mask.unsqueeze(-1)).as_complex()
            stft_masked = paddle.nn.functional.pad(stft_masked, (0,0,0,1025,0,0), 'constant')
            wav_masked = self._stft(stft_masked, inverse=True, length=source_audio.shape[1])
            if len(wav_masked.shape) == 3 and wav_masked.shape[-1] == 1:
                wav_masked = paddle.squeeze(wav_masked,-1,f'audio_{i}')
            if len(wav_masked.shape) == 4 and wav_masked.shape[-1] == 1:
                wav_masked = paddle.squeeze(wav_masked,-1,f'audio_{i}').squeeze(-1,f'audio_{i}')
            return_list.append(wav_masked.T)
            
        return *return_list,samplerate

if __name__ == '__main__':
    paddleaudio.backends.set_backend('soundfile') 
    sep = Separator(resume = ['model/net_vocal_0.pdparams', 'model/net_instrumental_0.pdparams'])
    sep.eval()
    # 对输入音频进行分离处理
    shuRu = list(paddleaudio.load('九万字-黄诗扶-60538547.flac'))
    data = paddle.to_tensor(librosa.resample(shuRu[0].numpy(), shuRu[1], 44100) )

    shuChu = sep(data,44100)
    for i in range(2):
            sf.write(f'测试输出{i}.wav', shuChu[i].detach().cpu().numpy().T, shuChu[-1], subtype='PCM_16')
