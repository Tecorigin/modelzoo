import numpy as np
import paddle.nn.functional as F
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window


from .constants import *


class STFT(paddle.nn.Layer):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length, hop_length, win_length=None, window='hann'):
        super(STFT, self).__init__()
        if win_length is None:
            win_length = filter_length

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = paddle.to_tensor(fourier_basis[:, None, :],dtype = paddle.float32)

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = paddle.to_tensor(fft_window).astype(paddle.float32)

            # window the bases
            forward_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.astype(paddle.float32))

    def forward(self, input_data):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[1]

        # similar to librosa, reflect-pad the input
        input_data = input_data.reshape([num_batches, 1, num_samples])
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            paddle.to_tensor(self.forward_basis,),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = paddle.sqrt(real_part**2 + imag_part**2)
        phase = paddle.to_tensor(paddle.atan2(imag_part.detach(), real_part.detach()),stop_gradient = False)

        return magnitude, phase


class MelSpectrogram(paddle.nn.Layer):
    def __init__(self, n_mels, sample_rate, filter_length, hop_length,
                 win_length=None, mel_fmin=0.0, mel_fmax=None):
        super(MelSpectrogram, self).__init__()
        self.stft = STFT(filter_length, hop_length, win_length)

        mel_basis = mel(sample_rate, filter_length, n_mels, mel_fmin, mel_fmax, htk=True)
        mel_basis = paddle.to_tensor(mel_basis).astype(paddle.float32)
        self.register_buffer('mel_basis', mel_basis)

    def forward(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, T, n_mels)
        """
        assert(float(paddle.min(y.detach())) >= -1),float(paddle.min(y.detach()))
        assert(float(paddle.max(y.detach())) <= 1),float(paddle.max(y.detach()))

        magnitudes, phases = self.stft(y)
        magnitudes = magnitudes.detach()
        mel_output = paddle.matmul(self.mel_basis, magnitudes)
        mel_output = paddle.log(paddle.clip(mel_output, min=1e-5))
        return mel_output


# the default melspectrogram converter across the project
melspectrogram = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)
#melspectrogram.to(DEFAULT_DEVICE)
