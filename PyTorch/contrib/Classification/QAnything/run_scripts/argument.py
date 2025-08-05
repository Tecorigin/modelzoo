# BSD 3-Clause License
# Copyright (c) 2023, Tecorigin Co., Ltd. All rights reserved.
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
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse

def parse_options():
    parser = argparse.ArgumentParser('QAnything embedding training', add_help=False)
    
    # 必需参数
    parser.add_argument('--train_file', type=str, required=True, 
                        help='Path to training data jsonl file')
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Local model directory path')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default="./outputs/emb_m3_sdaa",
                        help='Directory to save trained model and logs')
    parser.add_argument('--log_file', type=str, default="sdaa_loss.log",
                        help='Name of loss log file')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum training steps')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate for optimizer')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_amp', action="store_true",
                        help='Disable bfloat16 automatic mixed precision')
    
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    sys.exit(0)