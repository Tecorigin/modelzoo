#encoding=utf-8
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

from paddle import nn
from paddle.nn import Conv1D
import math
from functools import partial
from einops import rearrange, repeat

from .local_attention import LocalAttention
import paddle.nn.functional as F

'''
# for sdaa device,conv operator is not supported when group == 1.
class Conv1D(nn.Conv1D):
    def forward(self, x):
        padding = 0
        if self._padding_mode != "zeros":
            x = F.pad(
                x,
                self._reversed_padding_repeated_twice,
                mode=self._padding_mode,
                data_format=self._data_format,
            )
        else:
            padding = self._padding
        if self._groups != 1:
            out = F.conv1d(
                x.cpu(),
                self.weight.cpu(),
                bias=self.bias.cpu(),
                padding=padding,
                stride=self._stride,
                dilation=self._dilation,
                groups=self._groups,
                data_format=self._data_format,
            )
            return paddle.to_tensor(out)
        else:
            return F.conv1d(
                x,
                self.weight,
                bias=self.bias,
                padding=padding,
                stride=self._stride,
                dilation=self._dilation,
                groups=self._groups,
                data_format=self._data_format,
            )
'''

class ConformerNaiveEncoder(nn.Layer):
    """
    Conformer Naive Encoder

    Args:
        dim_model (int): Dimension of model
        num_layers (int): Number of layers
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
    """

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 dim_model: int,
                 use_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.use_norm = use_norm
        self.residual_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.attention_dropout = 0.1  # 废弃代码,仅做兼容性保留

        self.encoder_layers = nn.LayerList(
            [
                CFNEncoderLayer(dim_model, num_heads, use_norm, conv_only, conv_dropout, atten_dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None) -> paddle.Tensor:
        """
        Args:
            x (paddle.Tensor): Input tensor (#batch, length, dim_model)
            mask (paddle.Tensor): Mask tensor, default None
        return:
            paddle.Tensor: Output tensor (#batch, length, dim_model)
        """
        for (i, layer) in enumerate(self.encoder_layers):
            x = layer(x, mask)
        return x  # (#batch, length, dim_model)


class CFNEncoderLayer(nn.Layer):
    """
    Conformer Naive Encoder Layer

    Args:
        dim_model (int): Dimension of model
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.1
        atten_dropout (float): Dropout rate of attention module, default 0.1
    """

    def __init__(self,
                 dim_model: int,
                 num_heads: int = 8,
                 use_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.
                 ):
        super().__init__()

        if conv_dropout > 0.:
            self.conformer = nn.Sequential(
                ConformerConvModule(dim_model),
                nn.Dropout(conv_dropout)
            )
        else:
            self.conformer = ConformerConvModule(dim_model)
        self.norm = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(0.1)  # 废弃代码,仅做兼容性保留

        # selfatt -> fastatt: performer!
        if not conv_only:
            self.attn = SelfAttention(dim=dim_model,
                                      heads=num_heads,
                                      causal=False,
                                      use_norm=use_norm,
                                      dropout=atten_dropout, )
        else:
            self.attn = None

    def forward(self, x, mask=None) -> paddle.Tensor:
        """
        Args:
            x (paddle.Tensor): Input tensor (#batch, length, dim_model)
            mask (paddle.Tensor): Mask tensor, default None
        return:
            paddle.Tensor: Output tensor (#batch, length, dim_model)
        """
        if self.attn is not None:
            x = x + (self.attn(self.norm(x), mask=mask))
        x = x + (self.conformer(x))
        return x  # (#batch, length, dim_model)


class ConformerConvModule(nn.Layer):
    def __init__(
            self,
            dim,
            causal=False,
            expansion_factor=2,
            kernel_size=31,
            dropout=0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            Conv1D(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            Swish(),
            Conv1D(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DepthWiseConv1d(nn.Layer):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = Conv1D(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding, data_format="NCL")
        return self.conv(x)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Transpose(nn.Layer):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = list(dims)
        if self.dims[0] < self.dims[1]:
            self.dims[1],self.dims[0] = self.dims[0],self.dims[1]

    def forward(self, x):
        if x.dim() == 3:
            return x.transpose([0] + self.dims)
        elif x.dim() == 2:
            return x.transpose(self.dims)
        else:
            raise NotImplementedError()


class GLU(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, axis=self.dim)
        return out * paddle.nn.functional.sigmoid(gate)


class Swish(nn.Layer):
    def forward(self, x):
        return x * paddle.nn.functional.sigmoid(x)


class SelfAttention(nn.Layer):
    def __init__(self, dim, causal=False, heads=8, dim_head=64, local_heads=0, local_window_size=256, nb_features=None,
                 feature_redraw_interval=1000, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False,
                 dropout=0., no_projection=False, use_norm=False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal,
                                            generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                            qr_uniform_q=qr_uniform_q, no_projection=no_projection,
                                            use_norm=use_norm)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout,
                                         look_forward=int(not causal),
                                         rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None

        # print (heads, nb_features, dim_head)
        # name_embedding = paddle.zeros(110, heads, dim_head, dim_head)
        # self.name_embedding = nn.Parameter(name_embedding, requires_grad=True)

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @paddle.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()
        # paddle.nn.init.zeros_(self.name_embedding)
        # print (paddle.sum(self.name_embedding))

    def forward(self, x, context=None, mask=None, context_mask=None, name=None, inference=False, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        # print (paddle.sum(self.name_embedding))
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t.cpu()[:, :gh], t.cpu()[:, gh:]), (q, k, v))
        q, lq, k, lk, v, lv = map(lambda input:paddle.to_tensor(input),(q, lq, k, lk, v, lv)) # sdaa not support slice tensor to empty tensor.
        attn_outs = []
        # print (name)
        # print (self.name_embedding[name].size())
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)
            if cross_attend:
                pass
                # print (paddle.sum(self.name_embedding))
                # out = self.fast_attention(q,self.name_embedding[name],None)
                # print (paddle.sum(self.name_embedding[...,-1:]))
                # attn_outs.append(out)
            else:
                out = self.fast_attention(q, k, v)
                attn_outs.append(out)
        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)
        out = paddle.concat(attn_outs, axis=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class FastAttention(nn.Layer):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, generalized_attention=False,
                 kernel_fn=nn.ReLU(), qr_uniform_q=False, no_projection=False, use_norm=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features,
                                         nb_columns=dim_heads, scaling=ortho_scaling, qr_uniform_q=qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        self.use_norm = use_norm
        '''
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print(
                    'unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda
        '''
        if self.causal or self.generalized_attention:
            raise NotImplementedError('Causal and generalized attention not implemented yet')

    @paddle.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.place

        if self.use_norm:
            q = q / (q.norm(axis=-1, keepdim=True) + 1e-8)
            k = k / (k.norm(axis=-1, keepdim=True) + 1e-8)

        if self.no_projection:
            q = q.softmax(axis=-1)
            k = paddle.exp(k) if self.causal else k.softmax(axis=-2)

        elif self.generalized_attention:
            '''
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))
            '''
            raise NotImplementedError('generalized attention not implemented yet')

        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)

            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        if v is None:
            out = attn_fn(q, k, None)
            return out
        else:
            out = attn_fn(q, k, v)
            return out


def linear_attention(q, k, v):
    if v is None:
        # print (k.size(), q.size())
        out = paddle.einsum('...ed,...nd->...ne', k, q)
        return out

    else:
        k_cumsum = k.sum(axis=-2)
        # k_cumsum = k.sum(dim = -2)
        D_inv = 1. / (paddle.einsum('...nd,...d->...n', q, k_cumsum.astype(q.dtype)) + 1e-8)

        context = paddle.einsum('...nd,...ne->...de', k, v)
        # print ("TRUEEE: ", context.size(), q.size(), D_inv.size())
        out = paddle.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape
    # (batch size, head, length, model_dim)

    # normalize model dim
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    # what is ration?, projection_matrix.shape[0] --> 266

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.astype(data.dtype)

    # data_dash = w^T x
    data_dash = paddle.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    # diag_data = D**2
    diag_data = data ** 2
    diag_data = paddle.sum(diag_data, axis=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(axis=-1)

    # print ()
    if is_query:
        data_dash = ratio * (
                paddle.exp(data_dash - diag_data -
                          paddle.max(data_dash, axis=-1, keepdim=True)) + eps)#paddle.max(data_dash, axis=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            paddle.exp(data_dash - diag_data + eps))  # - paddle.max(data_dash)) + eps)

    return data_dash.astype(data.dtype)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    # print (nb_full_blocks)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q)
    # block_list[n] is a orthogonal matrix ... (model_dim * model_dim)
    # print (block_list[0].size(), paddle.einsum('...nd,...nd->...n', block_list[0], paddle.roll(block_list[0],1,1)))
    # print (nb_rows, nb_full_blocks, nb_columns)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    # print (remaining_rows)
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        # print (q[:remaining_rows].size())
        block_list.append(q[:remaining_rows])

    final_matrix = paddle.concat(block_list)

    if scaling == 0:
        multiplier = paddle.randn((nb_rows, nb_columns)).norm(axis=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * paddle.ones((nb_rows,))
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return paddle.diag(multiplier) @ final_matrix


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = paddle.randn((cols, cols))
    q, r = paddle.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: paddle.to_tensor(t,place=device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = paddle.diag(r, 0)
        q *= d.sign()
    return q.t()


def default(val, d):
    return val if exists(val) else d


def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0
