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

# Adapted to tecorigin hardware

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
import paddle.distributed as dist
_tcap_logger = None

def init_tcap_logger(name):
    
    global _tcap_logger
    _tcap_logger = Logger(
                [
                    StdOutBackend(Verbosity.DEFAULT),
                    JSONStreamBackend(Verbosity.VERBOSE, f"{name}_log.json"),
                ]
            )

    _tcap_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    _tcap_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
    _tcap_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
    _tcap_logger.metadata("val.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})
    _tcap_logger.metadata("train.compute_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    _tcap_logger.metadata("train.fp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    _tcap_logger.metadata("train.bp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    _tcap_logger.metadata("train.grad_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    _tcap_logger.metadata("summary.epoch", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    _tcap_logger.metadata("summary.metric", {"unit": "%", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    _tcap_logger.metadata("summary.best_epoch", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    _tcap_logger.metadata("summary.best_metric", {"unit": "%", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})

    return _tcap_logger

def log_at_trainer0(log):
    """
    logs will print multi-times when calling Fleet API.
    Only display single log and ignore the others.
    """

    def wrapper(fmt, *args):
        if dist.get_rank() == 0:
            log(fmt, *args)

    return wrapper


@log_at_trainer0
def info(fmt, *args):
    _tcap_logger.info(fmt, *args)
