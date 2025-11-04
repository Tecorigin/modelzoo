# encoding=gbk
import paddle
import numpy as np
import paddle.nn.functional as F
import os
import math
from paddle import Tensor
from typing import Optional, Union
import warnings
os.environ["LRU_CACHE_CAPACITY"] = "3"

def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
    dtype: Optional[paddle.dtype] = None,
):
    '''for onnx convert
    if paddle.in_dynamic_mode:
        try:
            if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
                raise Exception(
                    "Frequencies must be of integer type to ensure quality resampling computation. "
                    "To work around this, manually convert both frequencies to integer values "
                    "that maintain their resampling rate ratio before passing them into the function. "
                    "Example: To downsample a 44100 hz waveform by a factor of 8, use "
                    "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
                    "For more information, please refer to https://github.com/pytorch/audio/issues/1487."
                )
        except TypeError:
            pass
        '''
    assert paddle.to_tensor(orig_freq).astype(paddle.int32) == paddle.to_tensor(orig_freq)
    if resampling_method in ["sinc_interpolation", "kaiser_window"]:
        method_map = {
            "sinc_interpolation": "sinc_interp_hann",
            "kaiser_window": "sinc_interp_kaiser",
        }
        warnings.warn(
            f'"{resampling_method}" resampling method name is being deprecated and replaced by '
            f'"{method_map[resampling_method]}" in the next release. '
            "The default behavior remains unchanged."
        )
    elif resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = paddle.to_tensor(orig_freq,dtype=paddle.int32) // gcd
    new_freq = paddle.to_tensor(new_freq,dtype=paddle.int32) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= rolloff

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = paddle.ceil(paddle.to_tensor(lowpass_filter_width,dtype=paddle.float32) * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else paddle.float64

    idx = paddle.arange(-width, width + orig_freq, dtype=idx_dtype)[None, None] / orig_freq

    t = paddle.arange(0, -new_freq, -1, dtype=dtype)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clip_(-lowpass_filter_width, lowpass_filter_width)

    # we do not use built in torch windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    if resampling_method == "sinc_interp_hann":
        window = paddle.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = paddle.to_tensor(float(beta))
        window = paddle.i0(beta_tensor * paddle.sqrt(1 - (t / lowpass_filter_width) ** 2)) / paddle.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = paddle.where(t == 0, paddle.to_tensor(1.0).astype(t.dtype), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.astype(dtype=paddle.float32)

    return kernels, width

def _apply_sinc_resample_kernel(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: Tensor,
    width: int,
):
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")
    orig_freq = paddle.to_tensor(orig_freq,dtype=paddle.int32) // paddle.to_tensor(gcd,dtype=paddle.int32)
    new_freq = paddle.to_tensor(new_freq,dtype=paddle.int32) // paddle.to_tensor(gcd,dtype=paddle.int32)

    # pack batch
    shape = waveform.shape
    waveform = waveform.reshape([-1, shape[-1]])
    ori_dtype = waveform.dtype
    num_wavs, length = waveform.shape
    waveform = paddle.nn.functional.pad(waveform.unsqueeze(0), paddle.to_tensor((width, width + orig_freq),dtype=paddle.int32),data_format="NCL").squeeze(0)
    resampled = paddle.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq.item())
    resampled = resampled.transpose([0,2,1]).reshape([num_wavs, -1])
    target_length = (paddle.ceil(new_freq * length / orig_freq)).astype(paddle.int32)
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.reshpae([shape[:-1] + resampled.shape[-1:]])
    return resampled
class Resample(paddle.nn.Layer):
    r"""Resample a signal from one frequency to another. A resampling method can be given.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        If resampling on waveforms of higher precision than float32, there may be a small loss of precision
        because the kernel is cached once as float32. If high precision resampling is important for your application,
        the functional form will retain higher precision, but run slower because it does not cache the kernel.
        Alternatively, you could rewrite a transform that caches a higher precision kernel.

    Args:
        orig_freq (int, optional): The original frequency of the signal. (Default: ``16000``)
        new_freq (int, optional): The desired frequency. (Default: ``16000``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interp_hann``, ``sinc_interp_kaiser``] (Default: ``"sinc_interp_hann"``)
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        beta (float or None, optional): The shape parameter used for kaiser window.
        dtype (torch.device, optional):
            Determnines the precision that resampling kernel is pre-computed and cached. If not provided,
            kernel is computed with ``torch.float64`` then cached as ``torch.float32``.
            If you need higher precision, provide ``torch.float64``, and the pre-computed kernel is computed and
            cached as ``torch.float64``. If you use resample with lower precision, then instead of providing this
            providing this argument, please use ``Resample.to(dtype)``, so that the kernel generation is still
            carried out on ``torch.float64``.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.Resample(sample_rate, sample_rate/10)
        >>> waveform = transform(waveform)
    """

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        resampling_method: str = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: Optional[float] = None,
        *,
        dtype: Optional[paddle.dtype] = None,
    ) -> None:
        super().__init__()

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.gcd = paddle.gcd(self.orig_freq.astype(paddle.int32), paddle.to_tensor(self.new_freq))
        self.resampling_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta

        if self.orig_freq != self.new_freq:
            kernel, self.width = _get_sinc_resample_kernel(
                self.orig_freq,
                self.new_freq,
                self.gcd,
                self.lowpass_filter_width,
                self.rolloff,
                self.resampling_method,
                beta,
                dtype=dtype,
            )
            if paddle.in_dynamic_mode():
                self.register_buffer("kernel", kernel)
            else:
                self.kernel = kernel

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Output signal of dimension (..., time).
        """
        if self.orig_freq == self.new_freq:
            return waveform
        return _apply_sinc_resample_kernel(waveform, self.orig_freq, self.new_freq, self.gcd, self.kernel, self.width)

try:
    from librosa.filters import mel as librosa_mel_fn
except ImportError:
    print('  [INF0] torchfcpe.mel_tools.nv_mel_extractor: Librosa not found,'
          ' use torchfcpe.mel_tools.mel_fn_librosa instead.')
    from .mel_fn_librosa import mel as librosa_mel_fn


def dynamic_range_compression_paddle(x, C=1, clip_val=1e-5):
    return paddle.log(paddle.clip(x, min=clip_val) * C)


class HannWindow(paddle.nn.Layer):
    def __init__(self, win_size):
        super().__init__()
        window = paddle.audio.functional.get_window(window="hann", win_length=win_size, fftbins=True, dtype='float32')
        self.register_buffer('window', window, persistable=False)

    def forward(self):
        return self.window

class MelExtractor(paddle.nn.Layer):
    """Mel extractor

    Args:
        sr (int): Sampling rate. Defaults to 16000.
        n_mels (int): Number of mel bins. Defaults to 128.
        n_fft (int): FFT size. Defaults to 1024.
        win_size (int): Window size. Defaults to 1024.
        hop_length (int): Hop length. Defaults to 160.
        fmin (float, optional): min frequency. Defaults to 0.
        fmax (float, optional): max frequency. Defaults to sr/2.
        clip_val (float, optional): Clipping value. Defaults to 1e-5.
    """

    def __init__(self,
                 sr: Union[int, float],
                 n_mels: int,
                 n_fft: int,
                 win_size: int,
                 hop_length: int,
                 fmin: float = None,
                 fmax: float = None,
                 clip_val: float = 1e-5,
                 out_stft: bool = False
                 ):
        super().__init__()
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = sr / 2
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        #self.mel_basis = {}
        self.hann_window = None#{}

        mel = librosa_mel_fn(sr=self.target_sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        #self.mel_basis[mel_basis_key] = paddle.to_tensor(mel).astype(paddle.float32) # [mel_basis_key]
        self.register_buffer(
            'mel_basis',
            paddle.to_tensor((mel),dtype=paddle.float32),
            persistable=False
            )
        self.out_stft = out_stft
    r''' 旧代码
    @paddle.no_grad()
    def forward(self,
                 y: paddle.Tensor,  # (B, T, 1)
                 key_shift: Union[int, float] = 0,
                 speed: Union[int, float] = 1,
                 center: bool = False,
                 no_cache_window: bool = False
                 ) -> paddle.Tensor:  # (B, T, n_mels)
        """Get mel spectrogram

        Args:
            y (torch.Tensor): Input waveform, shape=(B, T, 1).
            key_shift (int, optional): Key shift. Defaults to 0.
            speed (int, optional): Variable speed enhancement factor. Defaults to 1.
            center (bool, optional): center for torch.stft. Defaults to False.
            no_cache_window (bool, optional): If True will clear cache. Defaults to False.
        return:
            spec (torch.Tensor): Mel spectrogram, shape=(B, T, n_mels).
        """

        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val

        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        if not no_cache_window:
            hann_window = self.hann_window
        else: # no_cache_window
            hann_window = {}

        y = y.squeeze(-1,name="y squeeze audio")
        if paddle.min(y) < -1.:
            print('[error with torchfcpe.mel_extractor.MelModule]min value is ', paddle.min(y))
        if paddle.max(y) > 1.:
            print('[error with torchfcpe.mel_extractor.MelModule]max value is ', paddle.max(y))
        mel_basis_key = str(fmax)

        if no_cache_window and (mel_basis_key not in mel_basis):
            mel = librosa_mel_fn(sr=self.target_sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
            mel_basis[mel_basis_key] = paddle.to_tensor(mel).astype(paddle.float32)
        
        key_shift_key = str(key_shift)
        if key_shift_key not in hann_window:
            hann_window[key_shift_key] = paddle.audio.functional.get_window(window="hann", win_length=win_size_new, fftbins=True, dtype='float32')#torch.hann_window(win_size_new) 

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.shape[-1] - pad_left)
        #mode = 'reflect' if pad_right < y.shape[-1] else 'constant'
        y = paddle.nn.functional.pad(y.unsqueeze(1), paddle.to_tensor((pad_left, pad_right),dtype=paddle.int32), mode='reflect' if pad_right < y.shape[-1] else 'constant', data_format="NCL")
        y = y.squeeze(1,"y squeeze 1")

        spec = paddle.signal.stft(y, n_fft=n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=hann_window[key_shift_key], center=center, pad_mode='reflect', normalized=False, onesided=True, name=None)
        spec = paddle.sqrt(spec.real().pow(2) + spec.imag().pow(2) + 1e-9,name = "spec")

        if key_shift != 0:
            size = n_fft // 2 + 1
            resize = spec.shape[1]
            pad_size = paddle.to_tensor(size - resize)
            pad_size = paddle.where(pad_size>=0,pad_size,paddle.to_tensor(0,dtype=pad_size.dtype))
            spec = F.pad(spec, paddle.to_tensor([0, 0, 0, size - resize],dtype=paddle.int32), data_format='NCL')
            spec = spec[:, :size, :] * win_size / win_size_new


        spec = paddle.matmul(mel_basis[mel_basis_key], spec,name="spec matmul") # 
        spec = dynamic_range_compression_paddle(spec, clip_val=clip_val)
        spec = spec.transpose([0,2,1])
        return spec  # (B, T, n_mels)
    '''

    @paddle.no_grad()
    def forward(self,
                 y: paddle.Tensor,  # (B, T, 1)
                 key_shift: Union[int, float] = 0,
                 speed: Union[int, float] = 1,
                 center: bool = False,
                 no_cache_window: bool = False
                 ) -> paddle.Tensor:  # (B, T, n_mels)
        """Get mel spectrogram

        Args:
            y (torch.Tensor): Input waveform, shape=(B, T, 1).
            key_shift (int, optional): Key shift. Defaults to 0.
            speed (int, optional): Variable speed enhancement factor. Defaults to 1.
            center (bool, optional): center for torch.stft. Defaults to False.
            no_cache_window (bool, optional): If True will clear cache. Defaults to False.
        return:
            spec (torch.Tensor): Mel spectrogram, shape=(B, T, n_mels).
        """

        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        clip_val = self.clip_val

        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))

        y = y.squeeze(-1)

        if paddle.min(y).item() < -1.:
            print('[error with torchfcpe.mel_extractor.MelModule]min value is ', paddle.min(y).item())
        if paddle.max(y).item() > 1.:
            print('[error with torchfcpe.mel_extractor.MelModule]max value is ', paddle.max(y).item())

        key_shift_key = str(key_shift)
        if not no_cache_window:
            if key_shift_key in self.hann_window:
                hann_window = self.hann_window[key_shift_key]
            else:
                hann_window = HannWindow(win_size_new).to(self.mel_basis.place)
                self.hann_window[key_shift_key] = hann_window
            hann_window_tensor = hann_window()
        else:
            hann_window_tensor = paddle.audio.functional.get_window(window="hann", win_length=win_size_new, fftbins=True, dtype='float32')            

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.shape[-1] - pad_left)

        y = paddle.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode='reflect' if pad_right < y.shape[-1] else 'constant', data_format="NCL")
        y = y.squeeze(1)

        spec = paddle.signal.stft(y, n_fft=n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=hann_window_tensor, center=center, pad_mode='reflect', normalized=False, onesided=True, name=None)
        spec = paddle.sqrt(spec.real().pow(2) + spec.imag().pow(2) + 1e-9, name = "spec")
        if key_shift != 0:
            size = n_fft // 2 + 1
            resize = spec.shape[1]
            if resize < size:
                spec = F.pad(spec.unsqueeze(0), paddle.to_tensor([0, 0, 0, size - resize],dtype=paddle.int32),).squeeze(0)
            spec = spec[:, :size, :] * win_size / win_size_new
        if self.out_stft:
            spec = spec[:, :512, :]
        else:
            #assert False,(self.mel_basis.shape,spec.shape) # [128, 513], [1, 414, 1260]
            spec = paddle.matmul(self.mel_basis, spec)
        spec = dynamic_range_compression_paddle(spec, clip_val=clip_val)
        spec = spec.transpose([0,2,1])
        return spec  # (B, T, n_mels)

# init nv_mel_extractor cache
mel_extractor = MelExtractor(16000, 128, 1024, 1024, 160, 0, 8000)


class Wav2Mel(paddle.nn.Layer):
    """
    Wav to mel converter

    Args:
        sr (int): Sampling rate. Defaults to 16000.
        n_mels (int): Number of mel bins. Defaults to 128.
        n_fft (int): FFT size. Defaults to 1024.
        win_size (int): Window size. Defaults to 1024.
        hop_length (int): Hop length. Defaults to 160.
        fmin (float, optional): min frequency. Defaults to 0.
        fmax (float, optional): max frequency. Defaults to sr/2.
        clip_val (float, optional): Clipping value. Defaults to 1e-5.
        device (str, optional): Device. Defaults to 'cpu'.
    """

    def __init__(self,
                 sr: Union[int, float],
                 n_mels: int,
                 n_fft: int,
                 win_size: int,
                 hop_length: int,
                 fmin: float = None,
                 fmax: float = None,
                 clip_val: float = 1e-5,
                 device='cpu'):
        super().__init__()
        # catch None
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = sr / 2
        # init
        self.sampling_rate = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.device = device
        self.resample_kernel = {}
        self.mel_extractor = MelExtractor(sr, n_mels, n_fft, win_size, hop_length, fmin, fmax, clip_val)

    def device(self):
        """Get device"""
        return self.parameters()[0].place

    @paddle.no_grad()
    def forward(self,
                 audio: paddle.Tensor,  # (B, T, 1)
                 sample_rate: Union[int, float],
                 keyshift: Union[int, float] = 0,
                 no_cache_window: bool = False
                 ) -> paddle.Tensor:  # (B, T, n_mels)
        """
        Get mel spectrogram

        Args:
            audio (torch.Tensor): Input waveform, shape=(B, T, 1).
            sample_rate (int): Sampling rate.
            keyshift (int, optional): Key shift. Defaults to 0.
            no_cache_window (bool, optional): If True will clear cache. Defaults to False.
        return:
            spec (torch.Tensor): Mel spectrogram, shape=(B, T, n_mels).
        """
        
        # resample
        if sample_rate == self.sampling_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate,
                    self.sampling_rate,
                    lowpass_filter_width=128
                )
            audio_res = self.resample_kernel[key_str](audio.squeeze(-1)).unsqueeze(-1)
        
        # extract
        mel = self.mel_extractor(audio_res, keyshift, no_cache_window=no_cache_window)
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = paddle.concat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]
        return mel  # (B, T, n_mels)


def unit_text():
    """
    Test unit for nv_mel_extractor.py
    Should be set path to your test audio file.
    Need matplotlib and librosa to plot.
    require: pip install matplotlib librosa
    """
    import time

    try:
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
    except ImportError:
        print('  [UNIT_TEST] torchfcpe.mel_tools.nv_mel_extractor: Matplotlib or Librosa not found,'
              ' skip plotting.')
        exit(1)

    # spawn mel extractor and wav2mel
    mel_extractor_test = MelExtractor(16000, 128, 1024, 1024, 160, 0, 8000)
    wav2mel_test = Wav2Mel(16000, 128, 1024, 1024, 160, 0, 8000)

    # load audio
    audio_path = r'test.wav'
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = paddle.to_tensor(audio).unsqueeze(0).unsqueeze(-1)
    print('  [UNIT_TEST] torchfcpe.mel_tools.nv_mel_extractor: Audio shape: {}'.format(audio.shape))

    # test mel extractor
    start_time = time.time()
    mel1 = mel_extractor_test(audio, 0, 1, False)
    print('  [UNIT_TEST] torchfcpe.mel_tools.nv_mel_extractor: Mel extractor time cost: {:.3f}s'.format(
        time.time() - start_time))
    print('  [UNIT_TEST] torchfcpe.mel_tools.nv_mel_extractor: Mel extractor output shape: {}'.format(mel1.shape))

    # test wav2mel
    start_time = time.time()
    mel2 = wav2mel_test(audio, 16000, 0)
    print('  [UNIT_TEST] torchfcpe.mel_tools.nv_mel_extractor: Wav2mel time cost: {:.3f}s'.format(
        time.time() - start_time))
    print('  [UNIT_TEST] torchfcpe.mel_tools.nv_mel_extractor: Wav2mel output shape: {}'.format(mel2.shape))

    # plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    librosa.display.waveshow(audio.squeeze().cpu().numpy(), sr=16000)
    plt.title('Audio')
    plt.subplot(1, 3, 2)
    librosa.display.specshow(mel1.squeeze().cpu().numpy().T, sr=16000, hop_length=160, x_axis='time', y_axis='mel')
    plt.title('Mel extractor')
    plt.subplot(1, 3, 3)
    librosa.display.specshow(mel2.squeeze().cpu().numpy().T, sr=16000, hop_length=160, x_axis='time', y_axis='mel')
    plt.title('Wav2mel')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    unit_text()
