## TCAP_DLLogger - minimal logging tool

本 log 工具是为了较为规范化的在模型训练/推断中打印和保存信息（在模型应用层），使用较为统一的格式以方便在后期进行文本解析和训练/推断结果分析。需要注意的是，分析整个训练/推断过程并不需要非常多的信息，最简洁的信息应该只包括每次迭代的 loss（或 accuracy） 和 speed，所以本工具的目标并不是事无巨细的记录训练/推断过程，而只是尽可能少的记录最核心的信息。

### Table Of Contents

- [TCAP\_DLLogger - minimal logging tool](#tcap_dllogger---minimal-logging-tool)
  - [Table Of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick Start Guide](#quick-start-guide)
    - [logger.log 记录每次迭代的关键信息](#loggerlog-记录每次迭代的关键信息)
    - [logger.metadata 记录关注指标的基本信息](#loggermetadata-记录关注指标的基本信息)
    - [logger.info 记录其他信息](#loggerinfo-记录其他信息)
  - [Available backends overview](#available-backends-overview)
  - [Example](#example)


### Installation

```sh
pip install git+https://gitee.com/xiwei777/tcap_dllogger.git
# or
git clone https://gitee.com/xiwei777/tcap_dllogger.git
cd tcap_dllogger
python setup.py install
```

### Quick Start Guide

在代码中加入以下几行代码，就可以使用该工具。

```python
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
    ]
)
```

以上代码表明该 logger 会打印输出到屏幕，并且也会写到名为 `tmp.json` 的 json 文件中。接下来，该 logger 支持三个方法：`logger.metadata()`,`logger.log()` 和 `logger.info()`.

#### logger.log 记录每次迭代的关键信息
一个简单的 `logger.log` 写法如下：

```python
logger.log(
                step = global_step,
                data = {"loss":_CE_loss.item(), "speed":ips},
                verbosity=Verbosity.DEFAULT,
            )
```
在上面的代码中:
- `global step` 为整个训练过程中的迭代次数，也可以写为 `step=[epoch_number, iteration_number]`。
- `data` 中以词典的形式记录数据，最重要的是 `loss` 和 `speed` 两个数据，其余数据可以自行添加记录，例如 `"compute_time=cpmpute_time"`, 对 key 和 value 值不做限制。

#### logger.metadata 记录关注指标的基本信息

`logger.metadata` 用来记录关注指标的信息，例如单位，格式等。写法是 `logger.metadata(metric_name, metric_metadata)`,`metric_metadata` 是一个字典。后台可以用 metadata 信息做 log。
简单的示例：
```python
logger.metadata("loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
logger.metadata(
    "speed",
    {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"},
)
```
#### logger.info 记录其他信息

`logger.info` 用来记录其他信息，例如命令行参数，或一些其他的信息。举例如下：

```python
args = parser.parse_args()
logger.info(data=args,)
logger.info(data="start training ...")
```

### Available backends overview

目前该工具支持两种后台输出，如前所示，包括输出到屏幕（StdOutBackend）和输出到 json 文件（JSONStreamBackend）。

### Example

```python
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

json_logger = Logger(
[
    StdOutBackend(Verbosity.DEFAULT),
    JSONStreamBackend(Verbosity.VERBOSE, 'dlloger_example.json'),
]
)

json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.loss_mean", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.compute_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.fp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.bp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.grad_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})

global_step = 0
for epo in range(4):
    # train_dataloader
    for idx in range(3):
        # make train
        pass

        json_logger.log(
            step = (epo, global_step),
            data = {
                    "rank":0,
                    "train.loss":0.01, 
                    "train.ips":3.0,
                    "data.shape":[16,3,224,224],
                    "train.lr":0.0001,
                    "train.data_time":1.03,
                    "train.compute_time":0.003,
                    "train.fp_time":3.02,
                    "train.bp_time":6.02,
                    "train.grad_time":0.003,
                    },
            verbosity=Verbosity.DEFAULT,
        )

        global_step += 1

    # val_loader
    for index in range(2):
        # make evaluation
        pass

    json_logger.log(
        step = (epo, global_step),
        data = {
            "val.loss":0.002,
            "val.ips":5.00,
            "val.top1":0.54
        },
        verbosity=Verbosity.DEFAULT,
    )
```
参数解析：
- rank：训练进程的 rank，单卡时设为 -1， ddp 时按照实际 rank 来；
- train.loss：模型训练时模型总的 loss；
- train.ips 或者 train.sps：指模型的吞吐量；
- data.shape：input data 的 shape；
- train.lr：学习率；
- train.data_time：从 dataloader 中取数据并 to device 的时间；
- train.compute_time：计算时间，包括前向+反向+优化器更新权重；
- train.fp_time：前向计算时间，包括计算 loss；
- train.bp_time：反向计算时间；
- train.grad_time：optimizer.step的时间；
- val.loss：验证集上的损失，如果没有可以设为 0；
- val.ips：或 val.sps验证集上的吞吐量；
- val.top1：模型评价指标，视模型而定。
