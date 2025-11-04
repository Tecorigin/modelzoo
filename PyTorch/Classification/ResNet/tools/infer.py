# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image
import argparse
import numpy as np
import json
import torch
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import torch_sdaa
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from image_classification.models import (
    resnet50,
    resnext101_32x4d,
    se_resnext101_32x4d,
)


def available_models():
    models = {
        m.name: m
        for m in [
            resnet50,
            resnext101_32x4d,
            se_resnext101_32x4d,
        ]
    }
    return models


def add_parser_arguments(parser):
    model_names = available_models().keys()
    parser.add_argument("--image-size", default="224", type=int)
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument(
        "--precision", metavar="PREC", default="AMP", choices=["AMP", "FP32"]
    )
    parser.add_argument("--cpu", action="store_true", help="perform inference on CPU")
    parser.add_argument("--image", metavar="<path>", help="path to classified image")


def load_jpeg_from_file(path, image_size, cuda=True):
    img_transforms = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.to('sdaa')
            std = std.to('sdaa')
            img = img.to('sdaa')
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


def check_quant_weight_correctness(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    quantizers_sd_keys = {
        f"{n[0]}._amax" for n in model.named_modules() if "quantizer" in n[0]
    }
    sd_all_keys = quantizers_sd_keys | set(model.state_dict().keys())
    assert set(state_dict.keys()) == sd_all_keys, (
        f"Passed quantized architecture, but following keys are missing in "
        f"checkpoint: {list(sd_all_keys - set(state_dict.keys()))}"
    )


def main(args, model_args):
    imgnet_classes = np.array(json.load(open("../docs/LOC_synset_mapping.json", "r")))


    model = resnet50(pretrained=True)

    if not args.cpu:
        model = model.to('sdaa')
    model.eval()

    input = load_jpeg_from_file(args.image, args.image_size, cuda=not args.cpu)

    with torch.no_grad(), autocast(enabled=args.precision == "AMP"):
        output = torch.nn.functional.softmax(model(input), dim=1)

    output = output.float().cpu().view(-1).numpy()
    top5 = np.argsort(output)[-5:][::-1]

    print(args.image)
    for c, v in zip(imgnet_classes[top5], output[top5]):
        print(f"{c}: {100*v:.1f}%")


def print_in_box(msg):
    print("#" * (len(msg) + 10))
    print(f"#### {msg} ####")
    print("#" * (len(msg) + 10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Classification")

    add_parser_arguments(parser)
    args, rest = parser.parse_known_args()
    model_args, rest = available_models()[args.arch].parser().parse_known_args(rest)

    assert len(rest) == 0, f"Unknown args passed: {rest}"

    cudnn.benchmark = True

    main(args, model_args)
