#BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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

import argparse

def parse_options():
    parser = argparse.ArgumentParser(description='ALM Example')

    parser.add_argument('--experiment', default='peta', type=str, required=True, help='(default=%(default)s)')
    parser.add_argument('--approach', default='inception_iccv', type=str, required=True, help='(default=%(default)s)')
    parser.add_argument('--epochs', default=60, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--optimizer', default='adam', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--weight_decay', default=0.0005, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--start-epoch', default=0, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--print_freq', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--save_freq', default=10, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--resume', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--decay_epoch', default=(20,40), type=eval, required=False, help='(default=%(default)d)')
    parser.add_argument('--prefix', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', required=False, help='evaluate model on validation set')
    
    return parser.parse_args()

if __name__ == "__main__":
    sys.exit(0)