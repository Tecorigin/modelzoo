#BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd.
# All rights reserved.
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
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import argparse

def parse_options():
    parser = argparse.ArgumentParser(description='MISF Face Inpainting Training Options')

    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--debug', type=int, default=1, help='Enable debug mode')
    parser.add_argument('--verbose', type=int, default=0, help='Enable verbose logging')

    parser.add_argument('--train_flist', type=str, default='./data/face.txt', help='Training image list file')
    parser.add_argument('--val_flist', type=str, default='./data/face.txt', help='Validation image list file')
    parser.add_argument('--test_flist', type=str, default='./data/face.txt', help='Testing image list file')

    parser.add_argument('--train_mask_flist', type=str, default='./data/mask.txt', help='Training mask list file')
    parser.add_argument('--val_mask_flist', type=str, default='./data/mask.txt', help='Validation mask list file')
    parser.add_argument('--test_mask_flist', type=str, default='./data/mask.txt', help='Testing mask list file')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--d2g_lr', type=float, default=0.1, help='Discriminator/Generator LR ratio')
    parser.add_argument('--beta1', type=float, default=0.0, help='Adam optimizer beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam optimizer beta2')

    parser.add_argument('--input_size', type=int, default=256, help='Input image size (0 for original)')
    parser.add_argument('--max_iters', type=int, default=100, help='Maximum training iterations')

    parser.add_argument('--l1_loss_weight', type=float, default=1.0, help='L1 loss weight')
    parser.add_argument('--fm_loss_weight', type=float, default=10.0, help='Feature-matching loss weight')
    parser.add_argument('--style_loss_weight', type=float, default=250.0, help='Style loss weight')
    parser.add_argument('--content_loss_weight', type=float, default=0.1, help='Content (perceptual) loss weight')
    parser.add_argument('--adv_loss_weight', type=float, default=0.1, help='Adversarial loss weight')

    parser.add_argument('--gan_loss', type=str, default='nsgan', help='GAN loss type: nsgan | lsgan | hinge')
    parser.add_argument('--gan_pool_size', type=int, default=0, help='Fake image pool size')

    parser.add_argument('--sample_interval', type=int, default=10, help='Sample interval')
    parser.add_argument('--sample_size', type=int, default=5, help='Sample image count per save')
    parser.add_argument('--log_interval', type=int, default=100000, help='Log interval')

    parser.add_argument('--mask_reverse', type=int, default=0, help='Reverse mask setting (for Dunhuang dataset)')
    parser.add_argument('--mask_threshold', type=int, default=0, help='Thresholding masks')

    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated GPU ids')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--save_interval', type=int, default=10000, help='Save model interval')
    parser.add_argument('--eval_interval', type=int, default=2000, help='Evaluate model interval')
    parser.add_argument('--train_sample_interval', type=int, default=1000, help='Train sample interval')
    parser.add_argument('--eval_sample_interval', type=int, default=200, help='Eval sample interval')

    parser.add_argument('--train_sample_save', type=str, default='./result/train_sample', help='Path to save training samples')
    parser.add_argument('--eval_sample_save', type=str, default='./result/eval_sample', help='Path to save evaluation samples')
    parser.add_argument('--test_sample_save', type=str, default='./result/test_20', help='Path to save test samples')

    parser.add_argument('--model_load', type=str, default='celebA_InpaintingModel', help='Name or path of model to load')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_options()
    print(args)
