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
import argparse
from collections import OrderedDict

def argparser(known_models):
    parser = argparse.ArgumentParser(description="Fine-tune BiT-M model.")
    parser.add_argument("--name", required=True,
                      help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument("--model", choices=list(known_models),
                      help="Which variant to use; BiT-M gives best results.")
    parser.add_argument("--logdir", required=True,
                      help="Where to log training info (small).")
    parser.add_argument("--bit_pretrained_dir", default='.',
                      help="Where to search for pretrained BiT models.")
    
    parser.add_argument("--dataset", choices=list(known_dataset_sizes.keys()),
                      help="Choose the dataset. It should be easy to add your own! "
                      "Don't forget to set --datadir if necessary.")
    parser.add_argument("--examples_per_class", type=int, default=None,
                      help="For the few-shot variant, use this many examples "
                      "per class only.")
    parser.add_argument("--examples_per_class_seed", type=int, default=0,
                      help="Random seed for selecting examples.")
    
    parser.add_argument("--batch", type=int, default=512,
                      help="Batch size.")
    parser.add_argument("--batch_split", type=int, default=1,
                      help="Number of batches to compute gradient on before updating weights.")
    parser.add_argument("--base_lr", type=float, default=0.003,
                      help="Base learning-rate for fine-tuning. Most likely default is best.")
    parser.add_argument("--eval_every", type=int, default=None,
                      help="Run prediction on validation set every so many steps."
                      "Will always run one evaluation at the end of training.")

    parser.add_argument("--datadir", required=True,
                  help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    return parser


KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])

known_dataset_sizes = {
  'cifar10': (32, 32),
  'cifar100': (32, 32),
  'oxford_iiit_pet': (224, 224),
  'oxford_flowers102': (224, 224),
  'imagenet2012': (224, 224),
}
