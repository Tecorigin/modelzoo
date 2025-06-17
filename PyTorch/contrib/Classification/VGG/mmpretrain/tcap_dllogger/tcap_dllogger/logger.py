# Copyright (c) 2019, Tecorigin CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import Namespace
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
import json
import atexit


class Backend(ABC):
    def __init__(self, verbosity):
        self._verbosity = verbosity

    @property
    def verbosity(self):
        return self._verbosity

    @abstractmethod
    def log(self, timestamp, elapsedtime, step, data):
        pass

    @abstractmethod
    def info(self, data):
        pass

    @abstractmethod
    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass


class Verbosity:
    OFF = -1
    DEFAULT = 0
    VERBOSE = 1


class Logger:
    def __init__(self, backends):
        self.backends = backends
        atexit.register(self.flush)
        self.starttime = datetime.now()

    def metadata(self, metric:str, metadata:dict):
        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            b.metadata(timestamp, elapsedtime, metric, metadata)

    def metadata_batch(self, metrics:list, metadata:dict):
        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            for metric in metrics:
                b.metadata(timestamp, elapsedtime, metric, metadata)

    def log(self, step, data, verbosity=1):
        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            if b.verbosity >= verbosity:
                b.log(timestamp, elapsedtime, step, data)

    def info(self, data,):
        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            b.info(timestamp, data)

    def flush(self):
        for b in self.backends:
            b.flush()


def default_step_format(step):
    return str(step)


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s


def default_metric_format(metric, metadata, value):
    unit = metadata["unit"] if "unit" in metadata.keys() else ""
    # format = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    if metric == "loss":
        metric = "train.loss"
    elif metric == "speed":
        metric = "train.speed"
    else:
        pass # str(float(value))

    if (value is not None and (metric != "rank") and not isinstance(value, (list, str))):
        stri = "{} : {} {}".format(metric, str(value), unit)
    else:
         stri = "{} : {} {}".format(metric, str(value), unit)
    # stri = "{} : {} {}".format(metric, format.format(str(value)) if (value is not None and (metric != "rank") and not isinstance(value, (list, str))) else value, unit)
    return stri

def default_prefix_format(rank, timestamp):
    return "TCAPPDLL {} - ".format(timestamp)

class StdOutBackend(Backend):
    def __init__(
        self,
        verbosity,
        step_format=format_step,
        metric_format=default_metric_format,
        prefix_format=default_prefix_format,
    ):
        super().__init__(verbosity=verbosity)

        self._metadata = defaultdict(dict)
        self.step_format = step_format
        self.metric_format = metric_format
        self.prefix_format = prefix_format
        if (os.getenv("WORLD_SIZE") is None and os.getenv("PADDLE_TRAINERS_NUM") is None):
            self.local_rank = 0
        elif (os.getenv("WORLD_SIZE") is not None and int(os.getenv("WORLD_SIZE")) == 1):
            self.local_rank = 0
        elif os.getenv("LOCAL_RANK") is not None:
            self.local_rank = int(os.getenv("LOCAL_RANK"))
        elif os.getenv("PADDLE_TRAINER_ID") is not None:
            self.local_rank = int(os.getenv("PADDLE_TRAINER_ID"))
        else:
            self.local_rank = 0

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        self._metadata[metric].update(metadata)

    def log(self, timestamp, elapsedtime, step, data):
        new_data = {}
        for key, value in data.items():
            if key == "rank" and not isinstance(value, str):
                new_data["rank"] = int(value)
            else:
                new_data[key] = value
        data = new_data
        print(
            "{}{} {}".format(
                self.prefix_format(self.local_rank, timestamp),
                self.step_format(step),
                " ".join(
                    [
                        self.metric_format(m, self._metadata[m], v)
                        for m, v in data.items()
                    ]
                ),
            )
        )

    def info(self,timestamp, data):
        print(data)

    def flush(self):
        pass


class JSONStreamBackend(Backend):
    def __init__(self, verbosity, filename, append=True):
        super().__init__(verbosity=verbosity)
        self._filename = filename
        self.file = open(filename, "a" if append else "w")
        atexit.register(self.file.close)
        if (os.getenv("WORLD_SIZE") is None and os.getenv("PADDLE_TRAINERS_NUM") is None):
            self.local_rank = 0
        elif (os.getenv("WORLD_SIZE") is not None and int(os.getenv("WORLD_SIZE")) == 1):
            self.local_rank = 0
        elif os.getenv("LOCAL_RANK") is not None:
            self.local_rank = int(os.getenv("LOCAL_RANK"))
        elif os.getenv("PADDLE_TRAINER_ID") is not None:
            self.local_rank = int(os.getenv("PADDLE_TRAINER_ID"))
        else:
            self.local_rank = 0
        # 清空 json
        if os.path.exists(filename):
            with open(filename, "r+") as f:
                f.seek(0)
                f.truncate()

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        self.file.write(
            "TCAPPDLL {}\n".format(
                json.dumps(
                    dict(
                        timestamp=str(timestamp.timestamp()),
                        elapsedtime=str(elapsedtime),
                        datetime=str(timestamp),
                        type="METADATA",
                        metric=metric,
                        metadata=metadata,
                    )
                )
            )
        )

    def log(self, timestamp, elapsedtime, step, data):
        new_data = {}
        for key, value in data.items():
            if key == "loss":
                new_data["QA.loss"] = value
            elif key == "speed":
                new_data["QA.speed"] = value
            elif key == "rank" and not isinstance(value, str):
                new_data["rank"] = int(value)
            else:
                new_data[key] = value
        data = {key:str(value) for key, value in new_data.items()}
        self.file.write(
            "TCAPPDLL {}\n".format(
                json.dumps(
                    dict(
                        timestamp=str(timestamp.timestamp()),
                        datetime=str(timestamp),
                        elapsedtime=str(elapsedtime),
                        type="LOG",
                        step=step,
                        data=str(data),
                    )
                )
            )
        )

    def info(self,timestamp, data):
        if isinstance(data, Namespace):
            data = data.__dict__
        self.file.write(
            "INFO {}\n".format(
                json.dumps(
                    dict(
                        local_rank=self.local_rank,
                        datetime=str(timestamp),
                        type="INFO",
                        data=data,
                    )
                )
            )
        )

    def flush(self):
        self.file.flush()
