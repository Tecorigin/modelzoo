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
import json
from .random import json_dict_to_random_boundary


def get_config():
    parser = argparse.ArgumentParser(description="Run Onnx Model Test")
    parser.add_argument('--json', type=str, help='config json file')
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable logging debug mode.",
    )
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s]: %(message)s',  \
            datefmt='%H:%M:%S')
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s]: %(message)s',  \
            datefmt='%H:%M:%S')
    logging.debug(f"json file: {args.json}")
    if args.json is None:
        raise ValueError("error: --json is empty")
    with open(args.json) as f:
        _test_configs = json.load(f)
        _boundary = _test_configs.get("boundary")
        if _boundary is not None:
            _test_configs["boundary"] = \
                json_dict_to_random_boundary(_boundary)
    case_name = args.json.split("/")[-1]
    return _test_configs, case_name
