import warnings
import os
import numpy as np
import paddle

def stft(
    x,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    pad_mode='reflect',
    normalized=False,
    onesided=True,
    name=None,
):
    r"""

    Short-time Fourier transform (STFT).

    The STFT computes the discrete Fourier transforms (DFT) of short overlapping
    windows of the input using this formula:
    
    .. math::
        X_t[\omega] = \sum_{n = 0}^{N-1}%
                      \text{window}[n]\ x[t \times H + n]\ %
                      e^{-{2 \pi j \omega n}/{N}}
    
    Where:
    - :math:`t`: The :math:`t`-th input window.

    - :math:`\omega`: Frequency :math:`0 \leq \omega < \text{n\_fft}` for `onesided=False`,
      or :math:`0 \leq \omega < \lfloor \text{n\_fft} / 2 \rfloor + 1` for `onesided=True`.

    - :math:`N`: Value of `n_fft`.

    - :math:`H`: Value of `hop_length`.

    Args:
        x (Tensor): The input data which is a 1-dimensional or 2-dimensional Tensor with
            shape `[..., seq_length]`. It can be a real-valued or a complex Tensor.
        n_fft (int): The number of input samples to perform Fourier transform.
        hop_length (int, optional): Number of steps to advance between adjacent windows
            and `0 < hop_length`. Default: `None`(treated as equal to `n_fft//4`)
        win_length (int, optional): The size of window. Default: `None`(treated as equal
            to `n_fft`)
        window (Tensor, optional): A 1-dimensional tensor of size `win_length`. It will
            be center padded to length `n_fft` if `win_length < n_fft`. Default: `None`(
            treated as a rectangle window with value equal to 1 of size `win_length`).
        center (bool, optional): Whether to pad `x` to make that the
            :math:`t \times hop\_length` at the center of :math:`t`-th frame. Default: `True`.
        pad_mode (str, optional): Choose padding pattern when `center` is `True`. See
            `paddle.nn.functional.pad` for all padding options. Default: `"reflect"`
        normalized (bool, optional): Control whether to scale the output by `1/sqrt(n_fft)`.
            Default: `False`
        onesided (bool, optional): Control whether to return half of the Fourier transform
            output that satisfies the conjugate symmetry condition when input is a real-valued
            tensor. It can not be `True` if input is a complex tensor. Default: `True`
        name (str, optional): The default value is None. Normally there is no need for user
            to set this property. For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        The complex STFT output tensor with shape `[..., n_fft//2 + 1, num_frames]`
        (real-valued input and `onesided` is `True`) or `[..., n_fft, num_frames]`
        (`onesided` is `False`)

    Examples:
        .. code-block:: python
    
            import paddle
            from paddle.signal import stft
    
            # real-valued input
            x = paddle.randn([8, 48000], dtype=paddle.float64)
            y1 = stft(x, n_fft=512)  # [8, 257, 376]
            y2 = stft(x, n_fft=512, onesided=False)  # [8, 512, 376]
    
            # complex input
            x = paddle.randn([8, 48000], dtype=paddle.float64) + \
                    paddle.randn([8, 48000], dtype=paddle.float64)*1j  # [8, 48000] complex128
            y1 = stft(x, n_fft=512, center=False, onesided=False)  # [8, 512, 372]

    """

    x_rank = len(x.shape)
    assert x_rank in [
        1,
        2,
    ], f'x should be a 1D or 2D real tensor, but got rank of x is {x_rank}'

    if x_rank == 1:  # (batch, seq_length)
        x = x.unsqueeze(0)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    assert hop_length > 0, f'hop_length should be > 0, but got {hop_length}.'

    if win_length is None:
        win_length = n_fft

    assert (
        0 < win_length <= n_fft
    ), f'win_length should be in (0, n_fft({n_fft})], but got {win_length}.'

    if window is not None:
        assert (
            len(window.shape) == 1 and window.shape[0] == win_length
        ), f'expected a 1D window tensor of size equal to win_length({win_length}), but got window with shape {window.shape}.'
    else:
        window = paddle.ones(shape=(win_length,), dtype=x.dtype)
        assert window.dtype == x.dtype

    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = paddle.nn.functional.pad(
            window, pad=[pad_left, pad_right], mode='constant'
        )

    if center:
        assert pad_mode in [
            'constant',
            'reflect',
        ], 'pad_mode should be "reflect" or "constant", but got "{}".'.format(
            pad_mode
        )

        pad_length = n_fft // 2
        # FIXME: Input `x` can be a complex tensor but pad does not supprt complex input.
        x = paddle.nn.functional.pad(
            x.unsqueeze(-1),
            pad=[pad_length, pad_length],
            mode=pad_mode,
            data_format="NLC",
        ).squeeze(-1)

    x_frames = paddle.signal.frame(x=x, frame_length=n_fft, hop_length=hop_length, axis=-1)
    x_frames = x_frames.transpose(
        perm=[0, 2, 1]
    )  # switch n_fft to last dim, egs: (batch, num_frames, n_fft)

    window = window.astype(x_frames.dtype)
    x_frames = paddle.multiply(x_frames, window)

    norm = 'ortho' if normalized else 'backward'
    if paddle.is_complex(x_frames):
        assert (
            not onesided
        ), 'onesided should be False when input or window is a complex Tensor.'

    if not paddle.is_complex(x):
        out = paddle.fft.fft_r2c(
            x=x_frames,
            n=None,
            axis=-1,
            norm=norm,
            forward=True,
            onesided=onesided,
            name=name,
        )
    else:
        out = paddle.fft.fft_c2c(
            x=x_frames, n=None, axis=-1, norm=norm, forward=True, name=name
        )

    out = out.transpose(perm=[0, 2, 1])  # (batch, n_fft, num_frames)

    if x_rank == 1:
        out.squeeze_(0)

    return out

def istft(
    x,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    normalized=False,
    onesided=True,
    length=None,
    return_complex=False,
    name=None,
):
    r"""
    Inverse short-time Fourier transform (ISTFT).

    Reconstruct time-domain signal from the giving complex input and window tensor when
        nonzero overlap-add (NOLA) condition is met:

    .. math::
        \sum_{t = -\infty}^{\infty}%
            \text{window}^2[n - t \times H]\ \neq \ 0, \ \text{for } all \ n

    Where:
    - :math:`t`: The :math:`t`-th input window.
    - :math:`N`: Value of `n_fft`.
    - :math:`H`: Value of `hop_length`.

    Result of `istft` expected to be the inverse of `paddle.signal.stft`, but it is
        not guaranteed to reconstruct a exactly realizible time-domain signal from a STFT
        complex tensor which has been modified (via masking or otherwise). Therefore, `istft`
        gives the [Griffin-Lim optimal estimate](https://ieeexplore.ieee.org/document/1164317)
        (optimal in a least-squares sense) for the corresponding signal.

    Args:
        x (Tensor): The input data which is a 2-dimensional or 3-dimensional **complesx**
            Tensor with shape `[..., n_fft, num_frames]`.
        n_fft (int): The size of Fourier transform.
        hop_length (int, optional): Number of steps to advance between adjacent windows
            from time-domain signal and `0 < hop_length < win_length`. Default: `None`(
            treated as equal to `n_fft//4`)
        win_length (int, optional): The size of window. Default: `None`(treated as equal
            to `n_fft`)
        window (Tensor, optional): A 1-dimensional tensor of size `win_length`. It will
            be center padded to length `n_fft` if `win_length < n_fft`. It should be a
            real-valued tensor if `return_complex` is False. Default: `None`(treated as
            a rectangle window with value equal to 1 of size `win_length`).
        center (bool, optional): It means that whether the time-domain signal has been
            center padded. Default: `True`.
        normalized (bool, optional): Control whether to scale the output by `1/sqrt(n_fft)`.
            Default: `False`
        onesided (bool, optional): It means that whether the input STFT tensor is a half
            of the conjugate symmetry STFT tensor transformed from a real-valued signal
            and `istft` will return a real-valued tensor when it is set to `True`.
            Default: `True`.
        length (int, optional): Specify the length of time-domain signal. Default: `None`(
            treated as the whole length of signal).
        return_complex (bool, optional): It means that whether the time-domain signal is
            real-valued. If `return_complex` is set to `True`, `onesided` should be set to
            `False` cause the output is complex.
        name (str, optional): The default value is None. Normally there is no need for user
            to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A tensor of least squares estimation of the reconstructed signal(s) with shape
            `[..., seq_length]`

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.signal import stft, istft

            paddle.seed(0)

            # STFT
            x = paddle.randn([8, 48000], dtype=paddle.float64)
            y = stft(x, n_fft=512)  # [8, 257, 376]

            # ISTFT
            x_ = istft(y, n_fft=512)  # [8, 48000]

            np.allclose(x, x_)  # True
    """

    x_rank = len(x.shape)
    assert x_rank in [
        2,
        3,
    ], 'x should be a 2D or 3D complex tensor, but got rank of x is {}'.format(
        x_rank
    )

    if x_rank == 2:  # (batch, n_fft, n_frames)
        x = x.unsqueeze(0)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    if win_length is None:
        win_length = n_fft

    # Assure no gaps between frames.
    assert (
        0 < hop_length <= win_length
    ), 'hop_length should be in (0, win_length({})], but got {}.'.format(
        win_length, hop_length
    )

    assert (
        0 < win_length <= n_fft
    ), 'win_length should be in (0, n_fft({})], but got {}.'.format(
        n_fft, win_length
    )

    n_frames = x.shape[-1]
    fft_size = x.shape[-2]

    if window is not None:
        assert (
            len(window.shape) == 1 and window.shape[0] == win_length
        ), 'expected a 1D window tensor of size equal to win_length({}), but got window with shape {}.'.format(
            win_length, window.shape
        )
    else:
        window_dtype = (
            paddle.float32
            if x.dtype in [paddle.float32, paddle.complex64]
            else paddle.float64
        )
        window = paddle.ones(shape=(win_length,), dtype=window_dtype)

    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        # FIXME: Input `window` can be a complex tensor but pad does not supprt complex input.
        window = paddle.nn.functional.pad(
            window, pad=[pad_left, pad_right], mode='constant'
        )

    x = x.transpose(
        perm=[0, 2, 1]
    )  # switch n_fft to last dim, egs: (batch, num_frames, n_fft)
    norm = 'ortho' if normalized else 'backward'


    if onesided is False:
        x = x[:, :, : n_fft // 2 + 1]
    out = paddle.fft.fft_c2r(x=x, n=None, axis=-1, norm=norm, forward=False, name=None)

    out = paddle.multiply(out, window).transpose(
        perm=[0, 2, 1]
    )  # (batch, n_fft, num_frames)
    out = paddle.signal.overlap_add(
        x=out, hop_length=hop_length, axis=-1
    )  # (batch, seq_length)

    # for export 自己加的
    def paddle_export_tile(x:paddle.Tensor,repeat_times:list):
        new = paddle.assign(x)
        unit = paddle.assign(x)
        for i in range(-1,-len(repeat_times)-1,-1):
            for j in range(repeat_times[i]-1):
                new = paddle.concat([new,unit],axis = i)
            unit = new.unsqueeze(0)
            new = new.unsqueeze(0)

        for i in range(len(x.shape)):
            unit = unit.squeeze(0)
        return unit

    window_envelop = paddle.signal.overlap_add(
        x=paddle_export_tile(
            x=paddle.multiply(window, window).unsqueeze(0),
            repeat_times=[n_frames, 1]
        ).transpose(
            perm=[1, 0]
        ),  # (n_fft, num_frames)
        hop_length=hop_length,
        axis=-1,
    )  # (seq_length, )

    if length is None:
        if center:
            out = out[:, (n_fft // 2) : -(n_fft // 2)]
            window_envelop = window_envelop[(n_fft // 2) : -(n_fft // 2)]
    else:
        if center:
            start = n_fft // 2
        else:
            start = 0

        out = paddle.slice(out,[1],[start],[start + length])
        #out = out[:, start : start + length]
        window_envelop = paddle.slice(window_envelop,[0],[start] ,[ start + length])

    out = paddle.divide(out , window_envelop)

    if x_rank == 2:
        out.squeeze_(0)

    return out

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

def deDaoZuiXinMoXing(params:HParams) -> list:
    moXinLuJing:str = params.model_dir
    moXing = list()
    epoch_list = list()
    for i in range(len(params['num_instruments'])):
        for j in os.listdir(moXinLuJing):
            if len(j.split('_')) == 3 and j.split('_')[0] == 'net' and j.split('_')[1] == params['num_instruments'][i] and j.split('_')[-1].endswith('.pdparams'):
                try:
                    model_epoch = int(j.split('_')[-1].replace('.pdparams',''))
                    epoch_list.append(model_epoch)

                except Exception as e:
                    continue
    epoch_list.sort()
    try:
        latest_epoch = epoch_list[-1]
    except:
        return [None,None]
    for meiGeYueQi in params['num_instruments']:
        if not os.path.isfile(os.path.join(params.model_dir, f'net_{meiGeYueQi}_{latest_epoch}.pdparams')):
            return [None, None]
        moXing.append(os.path.join(params.model_dir, f'net_{meiGeYueQi}_{latest_epoch}.pdparams'))

    return moXing
