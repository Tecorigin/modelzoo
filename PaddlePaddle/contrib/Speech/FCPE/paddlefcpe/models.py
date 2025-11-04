# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import weight_norm

from .model_conformer_naive import ConformerNaiveEncoder

class Transpose(nn.Layer):
    def __init__(self,perm:list) -> None:
        super().__init__()
        self.perm = perm
    def forward(self, x:paddle.Tensor) -> paddle.Tensor:
        return x.transpose(self.perm)

class Unsqueeze(nn.Layer):
    def __init__(self,axis:int) -> None:
        super().__init__()
        self.axis = axis
    def forward(self, x:paddle.Tensor) -> paddle.Tensor:
        return x.unsqueeze(self.axis)

class Squeeze(nn.Layer):
    def __init__(self,axis:int) -> None:
        super().__init__()
        self.axis = axis
    def forward(self, x:paddle.Tensor) -> paddle.Tensor:
        return x.squeeze(self.axis)

class CFNaiveMelPE(nn.Layer):
    """
    Conformer-based Mel-spectrogram Prediction Encoderc in Fast Context-based Pitch Estimation

    Args:
        input_channels (int): Number of input channels, should be same as the number of bins of mel-spectrogram.
        out_dims (int): Number of output dimensions, also class numbers.
        hidden_dims (int): Number of hidden dimensions.
        n_layers (int): Number of conformer layers.
        f0_max (float): Maximum frequency of f0.
        f0_min (float): Minimum frequency of f0.
        use_fa_norm (bool): Whether to use fast attention norm, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
    """

    def __init__(self,
                 input_channels: int,
                 out_dims: int,
                 hidden_dims: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 f0_max: float = 1975.5,
                 f0_min: float = 32.70,
                 use_fa_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.use_fa_norm = use_fa_norm
        self.residual_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.attention_dropout = 0.1  # 废弃代码,仅做兼容性保留

        # Input stack, convert mel-spectrogram to hidden_dims
        self.input_stack = nn.Sequential(
            Transpose([0, 2, 1]),
            nn.Conv1D(input_channels, hidden_dims, 3, 1, 1),
            Unsqueeze(-2),
            nn.GroupNorm(4, hidden_dims),
            Squeeze(-2),
            nn.LeakyReLU(),
            nn.Conv1D(hidden_dims, hidden_dims, 3, 1, 1),
            Transpose([0, 2, 1]),
        )
        # Conformer Encoder
        self.net = ConformerNaiveEncoder(
            num_layers=n_layers,
            num_heads=n_heads,
            dim_model=hidden_dims,
            use_norm=use_fa_norm,
            conv_only=conv_only,
            conv_dropout=conv_dropout,
            atten_dropout=atten_dropout
        )
        # LayerNorm
        self.norm = nn.LayerNorm(hidden_dims)
        # Output stack, convert hidden_dims to out_dims
        self.output_proj = weight_norm(
            nn.Linear(hidden_dims, out_dims),
            'weight',
            1
        )
        # Cent table buffer
        """
        self.cent_table_b = paddle.Tensor(
            np.linspace(self.f0_to_cent(paddle.Tensor([f0_min]))[0], self.f0_to_cent(paddle.Tensor([f0_max]))[0],
                        out_dims))
        """

        self.cent_table_b = paddle.linspace(self.f0_to_cent(paddle.to_tensor([self.f0_min]))[0],
                                           self.f0_to_cent(paddle.to_tensor([self.f0_max]))[0],
                                           self.out_dims)
        self.register_buffer("cent_table", self.cent_table_b)

        # gaussian_blurred_cent_mask_b buffer
        self.gaussian_blurred_cent_mask_b = ((1200. * paddle.log2(paddle.to_tensor([self.f0_max / 10.])))[0].detach()).reshape([1])
        self.register_buffer("gaussian_blurred_cent_mask", self.gaussian_blurred_cent_mask_b)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            x (paddle.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
        return:
            paddle.Tensor: Predicted f0 latent, shape (B, T, out_dims).
        """
        x = self.input_stack(x) # GroupNorm, NCL -> NCHW -> GroupNorm -> NCHW  -> NCL
        x = self.net(x)
        x = self.norm(x)
        x = self.output_proj(x)
        x = paddle.nn.functional.sigmoid(x)
        return x  # latent (B, T, out_dims)

    @paddle.no_grad()
    def latent2cents_decoder(self,
                             y: paddle.Tensor,
                             threshold: float = 0.05,
                             mask: bool = True
                             ) -> paddle.Tensor:
        """
        Convert latent to cents.
        Args:
            y (paddle.Tensor): Latent, shape (B, T, out_dims).
            threshold (float): Threshold to mask. Default: 0.05.
            mask (bool): Whether to mask. Default: True.
        return:
            paddle.Tensor: Cents, shape (B, T, 1).
        """
        B, N, _ = y.shape

        # for paddlepaddle2.4 ONNX export
        cent_table = paddle.linspace(self.f0_to_cent(paddle.to_tensor([self.f0_min]))[0],
                                           self.f0_to_cent(paddle.to_tensor([self.f0_max]))[0],
                                           self.out_dims)
        ci = cent_table[None, None, :].expand([B, N, -1])
        rtn = paddle.sum(ci * y, axis=-1, keepdim=True) / paddle.sum(y, axis=-1, keepdim=True)  # cents: [B,N,1]
        if mask:
            confident = paddle.max(y, axis=-1, keepdim=True)[0]
            confident_mask = paddle.ones_like(confident)
            #confident_mask[confident <= threshold] = float("-INF")
            confident_mask = paddle.where(confident <= threshold,\
                paddle.to_tensor(float("-INF"),dtype=confident_mask.dtype),\
                confident_mask)
            rtn = rtn * confident_mask
        return rtn  # (B, T, 1)

    @paddle.no_grad()
    def latent2cents_local_decoder(self,
                                   y: paddle.Tensor,
                                   threshold: float = 0.05,
                                   mask: bool = True
                                   ) -> paddle.Tensor:
        """
        Convert latent to cents. Use local argmax.
        Args:
            y (paddle.Tensor): Latent, shape (B, T, out_dims).
            threshold (float): Threshold to mask. Default: 0.05.
            mask (bool): Whether to mask. Default: True.
        return:
            paddle.Tensor: Cents, shape (B, T, 1).
        """
        B, N, _ = y.shape
        # for paddlepaddle2.4 ONNX export
        cent_table = paddle.linspace(self.f0_to_cent(paddle.to_tensor([self.f0_min]))[0],
                                           self.f0_to_cent(paddle.to_tensor([self.f0_max]))[0],
                                           self.out_dims)
        ci = cent_table[None, None, :].expand([B, N, -1])
        confident, max_index = paddle.max(y, axis=-1, keepdim=True), paddle.argmax(y, axis=-1, keepdim=True)
        local_argmax_index = paddle.arange(0, 9) + (max_index - 4)
        #local_argmax_index[local_argmax_index < 0] = 0 # for ONNX export
        local_argmax_index = paddle.where(local_argmax_index < 0,\
            paddle.to_tensor(0,dtype=local_argmax_index.dtype),\
            local_argmax_index)
        #local_argmax_index[local_argmax_index >= self.out_dims] = self.out_dims - 1 # for ONNX export
        local_argmax_index = paddle.where(local_argmax_index >= self.out_dims,\
            paddle.to_tensor(self.out_dims - 1,dtype=local_argmax_index.dtype),\
            local_argmax_index)
        ci_l = paddle.take_along_axis(ci, axis=-1, indices=local_argmax_index)
        y_l = paddle.take_along_axis(y, axis=-1, indices=local_argmax_index)
        rtn = paddle.sum(ci_l * y_l, axis=-1, keepdim=True) / paddle.sum(y_l, axis=-1, keepdim=True)  # cents: [B,N,1]
        if mask:
            confident_mask = paddle.ones_like(confident)
            #confident_mask[confident <= threshold] = paddle.to_tensor(float("-INF"))
            confident_mask = paddle.where(confident <= threshold,\
                paddle.to_tensor(float("-INF"),dtype=confident_mask.dtype),\
                confident_mask)
            rtn = rtn * confident_mask
        return rtn  # (B, T, 1)

    @paddle.no_grad()
    def gaussian_blurred_cent2latent(self, cents):  # cents: [B,N,1]
        """
        Convert cents to latent.
        Args:
            cents (paddle.Tensor): Cents, shape (B, T, 1).
        return:
            paddle.Tensor: Latent, shape (B, T, out_dims).
        """
        mask = paddle.logical_and((cents > 0.1), (cents < self.gaussian_blurred_cent_mask))
        # mask = (cents>0.1) & (cents<(1200.*np.log2(self.f0_max/10.)))
        B, N, _ = cents.shape
        ci = self.cent_table[None, None, :].expand([B, N, -1])
        return paddle.exp(-paddle.square(ci - cents) / 1250) * mask.astype(paddle.float32)

    @paddle.no_grad()
    def infer(self,
              mel: paddle.Tensor,
              decoder: str = "local_argmax",  # "argmax" or "local_argmax"
              threshold: float = 0.05,
              ) -> paddle.Tensor:
        """
        Args:
            mel (paddle.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
            decoder (str): Decoder type. Default: "local_argmax".
            threshold (float): Threshold to mask. Default: 0.05.
        """
        latent = self.forward(mel)
        if decoder == "argmax":
            cents = self.latent2cents_decoder(latent, threshold=threshold)
        elif decoder == "local_argmax":
            cents = self.latent2cents_local_decoder(latent, threshold=threshold)
        else:
            raise ValueError(f"  [x] Unknown decoder type {decoder}.")
        f0 = self.cent_to_f0(cents)
        return f0  # (B, T, 1)

    def train_and_loss(self, mel, gt_f0, loss_scale=10):
        """
        Args:
            mel (paddle.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
            gt_f0 (paddle.Tensor): Ground truth f0, shape (B, T, 1).
            loss_scale (float): Loss scale. Default: 10.
        return: loss
        """
        if mel.shape[-2] != gt_f0.shape[-2]:
            _len = min(mel.shape[-2], gt_f0.shape[-2])
            mel = mel[:, :_len, :]
            gt_f0 = gt_f0[:, :_len, :]
        gt_cent_f0 = self.f0_to_cent(gt_f0)  # mel f0, [B,N,1]
        x_gt = self.gaussian_blurred_cent2latent(gt_cent_f0)  # [B,N,out_dim]
        x = self.forward(mel)  # [B,N,out_dim]
        loss = F.binary_cross_entropy(x, x_gt) * loss_scale
        return loss

    @paddle.no_grad()
    def cent_to_f0(self, cent: paddle.Tensor) -> paddle.Tensor:
        """
        Convert cent to f0. Args: cent (paddle.Tensor): Cent, shape = (B, T, 1). return: paddle.Tensor: f0, shape = (B, T, 1).
        """
        f0 = 10. * 2 ** (cent / 1200.)
        return f0  # (B, T, 1)

    @paddle.no_grad()
    def f0_to_cent(self, f0: paddle.Tensor) -> paddle.Tensor:
        """
        Convert f0 to cent. Args: f0 (paddle.Tensor): f0, shape = (B, T, 1). return: paddle.Tensor: Cent, shape = (B, T, 1).
        """
        cent = 1200. * paddle.log2(f0 / 10.)
        return cent  # (B, T, 1)
