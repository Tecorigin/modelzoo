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
import torchvision
from pathlib import Path
import os
import glob
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from functools import partial
import numpy as np
from torchvision.transforms.functional import InterpolationMode

RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1)) 

def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    return imgs, targets

def load_data(valdir, batch_size,rank=-1):
    # Data loading code
    print("Loading data")


    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
    )

    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', rank))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 0))
    
    print("Creating data loaders")
    if rank== -1:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False,num_replicas=world_size, rank=rank)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True,
        collate_fn=partial(fast_collate, torch.contiguous_format),shuffle=(test_sampler is None),
        drop_last=True ,
    )
    
    return data_loader_test

