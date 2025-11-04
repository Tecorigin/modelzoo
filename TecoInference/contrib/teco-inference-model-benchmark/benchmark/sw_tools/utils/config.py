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
import logging
import yaml


def get_config():
    parser = argparse.ArgumentParser(description="Get special tensor diffs.")
    parser.add_argument(
        "--case_path",
        type=str,
        required=False,
        help="the path of json file.",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="test network or op.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        help="enable performance or precion mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable logging debug mode.",
    )
    parser.add_argument(
        "-i",
        "--images_path",
        type=str,
        help="path to one image  or file including images.",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        help="path to a model to load.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU, SW",
        help="select device to run.",
    )
    parser.add_argument(
        "-api",
        type=str,
        default="async",
        help="using sync or async. default async",
    )
    parser.add_argument(
        "-batch",
        type=str,
        required=False,
        help="batch_size of input to infer.",
    )
    parser.add_argument(
        "-shape",
        type=str,
        required=False,
        help="input shape of input to infer.",
    )
    parser.add_argument(
        "-dump_path",
        type=str,
        default="./dump_files",
        help="specalize a path to store debug data.",
    )
    parser.add_argument("-result_path",
                        type=str,
                        default="./benchmark_result",
                        help="specalize a path to store results.")
    opt = parser.parse_args()
    if opt.debug:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d]' \
                                    '[%(levelname)s]: %(message)s', \
                            datefmt='%H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d]' \
                                    '[%(levelname)s]: %(message)s', \
                            datefmt='%H:%M:%S')

    case_info = {}
    logging.info("[Step 1/6] Parsing and validating input arguments")
    if opt.case_path is not None:
        with open(opt.case_path, 'r') as f:
            # case_info["input_case"] = json.load(f)
            case_info["input_case"] = yaml.load(f, Loader=yaml.FullLoader)
            logging.info(f"Use {opt.case_path} to achieve config info.")
    else:
        case_info["batch"] = opt.batch
        case_info["input_shape"] = opt.shape
        case_info["input_path"] = opt.input
        case_info["model_path"] = opt.model_path

    # Command line priority higher
    for key in case_info["input_case"]:
        if "images_path" in case_info["input_case"][key] and opt.images_path is not None:
            case_info["input_case"][key]["images_path"] = opt.images_path
        if "model_path" in case_info["input_case"][key] and opt.model_path is not None:
            case_info["input_case"][key]["model_path"] = opt.model_path
    case_info["api"] = opt.api
    case_info["type"] = opt.type
    case_info["mode"] = opt.mode
    case_info["device"] = opt.device
    case_info["result_path"] = opt.result_path
    return case_info
