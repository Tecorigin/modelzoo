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

import torch
import torch.nn as nn
import math
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_length, d_model)   # here, pe.size = [1500, 64]
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)  # here, position,size = [1500, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x * torch.sqrt(self.d_model)
        x = x + self.pe
        return x

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout, max_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(10, d_model),
            nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        # self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, 256, dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, 10)

    def forward(self, src, tgt=None):
        
        src = self.embedding(src)

        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)
        
        output = self.output_layer(output)

        return output


def get_model(args):
    return TransformerModel(args.d_model, args.n_head, args.num_layers, args.dropout, args.max_length)


if __name__ == '__main__':
    input = torch.rand([15, 100, 1])
    transformer = TransformerModel(128, 8, 2, 0.1, 100)
    output = transformer(input)
    print(output.size())
