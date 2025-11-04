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
import paddle.nn.functional as F
from paddle import Tensor
from einops import rearrange

from .local_attention import LocalAttention

# helper function
def Parameter(data:Tensor,requires_grad:bool=True):
    y = paddle.create_parameter(shape=data.shape,
                            dtype=data.dtype,
                            default_initializer=paddle.nn.initializer.Assign(data))
    y.stop_gradient = not requires_grad
    return y

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling functions

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = paddle.topk(logits, k)
    probs = paddle.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# multi-head attention

class LocalMHA(nn.Layer):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        prenorm = False,
        qk_rmsnorm = False,
        qk_scale = 8,
        use_xpos = False,
        xpos_scale_base = None,
        exact_windowsize = None,
        gate_values_per_head = False,
        **kwargs
    ):
        super().__init__()        
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(paddle.ones(dim_head))
            self.k_scale = nn.Parameter(paddle.ones(dim_head))

        self.attn_fn = LocalAttention(
            dim = dim_head,
            window_size = window_size,
            causal = causal,
            autopad = True,
            scale = (qk_scale if qk_rmsnorm else None),
            exact_windowsize = default(exact_windowsize, True),
            use_xpos = use_xpos,
            xpos_scale_base = xpos_scale_base,
            **kwargs
        )

        self.to_v_gate = None

        if gate_values_per_head:
            self.to_v_gate = nn.Sequential(
                nn.Linear(dim, heads)
            )

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None, attn_bias = None):
        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) 

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        out = self.attn_fn(q, k, v, mask = mask, attn_bias = attn_bias)

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            gates = rearrange(gates, 'b n h -> b h n 1')
            out = out * gates.sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# feedforward

class GEGLU(nn.Layer):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# dynamic positional bias

class DynamicPositionBias(nn.Layer):
    def __init__(
        self,
        dim,
        heads
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, heads)
        )

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, i, j):
        device = self.device
        assert j >= i

        rel_dist = paddle.arange(j, dtype = paddle.float32)
        bias = self.mlp(rearrange(rel_dist, '... -> ... 1'))

        i_seq = paddle.arange(j - i, j)
        j_seq = paddle.arange(j)

        rel_dist_indices = (rearrange(i_seq, 'i -> i 1') - rearrange(j_seq, 'j -> 1 j')).abs()

        bias = rearrange(bias[rel_dist_indices], 'i j h -> h i j')
        return bias

# main transformer class

class LocalTransformer(nn.Layer):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        causal = True,
        local_attn_window_size = 512,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ignore_index = -1,
        use_xpos = False,
        xpos_scale_base = None,
        use_dynamic_pos_bias = False,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len
        self.layers = nn.LayerList([])

        self.local_attn_window_size = local_attn_window_size
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        for _ in range(depth):
            self.layers.append(nn.LayerList([
                LocalMHA(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = causal, window_size = local_attn_window_size, use_xpos = use_xpos, xpos_scale_base = xpos_scale_base, use_rotary_pos_emb = not use_dynamic_pos_bias, prenorm = True, **kwargs),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.ignore_index = ignore_index
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    @paddle.no_grad()
    @eval_decorator
    def generate(
        self,
        prime,
        seq_len,
        temperature = 1.,
        filter_thres = 0.9,
        **kwargs
    ):
        n, device = prime.shape[1], prime.device

        out = prime

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.max_seq_len:], **kwargs)
            filtered_logits = top_k(logits[:, -1], thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sampled = paddle.multinomial(probs, 1)
            out = paddle.concat((out, sampled), axis = -1)

        return out[:, n:]

    def forward(self, x, mask = None, return_loss = False):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        assert n <= self.max_seq_len
        x = x + self.pos_emb(paddle.arange(n, device = device))

        # dynamic pos bias

        attn_bias = None
        if exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        # go through layers

        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_bias = attn_bias) + x
            x = ff(x) + x

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        loss = F.cross_entropy(logits, labels, ignore_index = self.ignore_index)
        return loss
