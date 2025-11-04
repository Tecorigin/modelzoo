## dump_tools.py 参数及使用说明

| 参数名称 | 参数说明|
| --- | :---: |
|dump_full|必选, 用来打印对应的网络逐层数据|
|dump_once|必选, 用来打印网络中某一层的数据|
|dump_range|必选。用来打印某一段网络的逐层数据|

## 当主参数为 dump_full 时，子参数说明如下：

| 参数名称 | 参数说明|
| --- | :---: |
|frame_type|必选。需要跑的框架类型：sdaa（神威框架）， ort（onnxruntime框架）|
|onnx_dir|必选。onnx模型路径|
|model_type|可选。onnx模型的类型，默认为fp16。|
|step|可选。默认要dump的层数，默认是1，若要每间隔10层打印，step可设置为10|
|input_file|可选。输入的数据路径，形式为--input key:value key=input_name, value=input_data.npy|
|passed_path|可选。仅在sdaa模式下可用，该模型下需要跑的pass的py文件， 若不设置会走默认的passed文件|
|compare_dir|可选。需要比较的数据集合的路径。|
|golden_dir|可选。需要比较的 golden 数据集合的路径。|
|input_tag|可选。仅在sdaa模式下可用， 推理数据与 golden 数据相差的倍数，默认为10|
|th1|可选。sdaa 推理的精度阈值，默认为0.000001。|
|th2|可选。diff2比较误差阈值，默认为0.001。|

一个具体的使用示例如下：

```
python dump_tools dump_full -f sdaa -d path/to/model_sample.onnx \
                            -s 20 \
                            -cd /path/to/compare_npy_file/ \
                            -g /path/to/golden_npy_file/ \
                            -t 5 \
                            -t1 0.02
```

文件对比结果如下
```
index: 0 | topoidx_00000__graph_input_cast_0.npy | nan_inf: False | diff2: 7.600144662230223e-07 | diff3_max=fail | diff3_mean=fail | diff3_max_sw: 5.060434341430664e-05 | diff3_max_g: 0.0*5.0=0.0 | diff3_mean_sw: 4.710508871852426e-09 | diff3_mean_g: 0.0*5.0=0.0
```

## 当主参数为 dump_once 时，子参数说明如下：

| 参数名称 | 参数说明|
| --- | :---: |
|frame_type|必选。需要跑的框架类型：sdaa（神威框架）， ort（onnxruntime框架）|
|onnx_dir|必选。onnx模型路径|
|model_type|可选。onnx模型的类型，默认为fp16。|
|output_name|必选。要打印的 output 的name|
|input_file|可选。输入的数据路径，形式为--input key:value key=input_name, value=input_data.npy|
|passed_path|可选。仅在sdaa模式下可用，该模型下需要跑的pass的py文件, 若不设置会走默认的passed文件|
|compare_dir|可选。需要比较的数据集合的路径。|
|golden_dir|可选。需要比较的 golden 数据集合的路径。|
|input_tag|可选。仅在sdaa模式下可用， 推理数据与 golden 数据相差的倍数，默认为10|
|th1|可选。sdaa 推理的精度阈值，默认为0.000001。|
|th2|可选。diff2比较误差阈值，默认为0.001。|

一个具体的使用示例如下：

```
python dump_tools dump_once -f ort -d path/to/model_sample.onnx \
                            -n relu_3.tmp_0 \
                            -cd /path/to/compare_npy_file/ \
                            -g /path/to/golden_npy_file/ \
                            -t 10 \
                            -t1 0.02
```

文件对比结果如下
```
index: 0 | relu_3.tmp_0.npy | nan_inf: False | diff2: 0.0 | diff3_max=pass | diff3_mean=pass | diff3_max_sw: 0.0 | diff3_max_g: 0.0*10.0=0.0 | diff3_mean_sw: 0.0 | diff3_mean_g: 0.0*10.0=0.0
```

## 当主参数为 dump_range 时，子参数说明如下：

| 参数名称 | 参数说明|
| --- | :---: |
|frame_type|必选。需要跑的框架类型：sdaa（神威框架）， ort（onnxruntime框架）|
|onnx_dir|必选。onnx模型路径|
|model_type|可选。onnx模型的类型，默认为fp16。|
|begin|可选。要打印 onnx 模型层数的开始的 topo 位置|
|begin|可选。要打印 onnx 模型层数的结束的 topo 位置|
|begin_string|可选。要打印 onnx 模型开始的 output name 的名字|
|end_string|可选。要打印 onnx 模型结束的 output name 的名字|
|step|可选。默认要dump的层数，默认是1，若要每间隔10层打印，step可设置为10|
|input_file|可选。输入的数据路径，形式为--input key:value key=input_name, value=input_data.npy|
|passed_path|可选。仅在sdaa模式下可用，该模型下需要跑的pass的py文件, 若不设置会走默认的passed文件|
|compare_dir|可选。需要比较的数据集合的路径。|
|golden_dir|可选。需要比较的 golden 数据集合的路径。|
|input_tag|可选。仅在sdaa模式下可用， 推理数据与 golden 数据相差的倍数，默认为10|
|th1|可选。sdaa 推理的精度阈值，默认为0.000001。|
|th2|可选。diff2比较误差阈值，默认为0.001。|

一个具体的使用示例如下：

```
python dump_tools dump_range -f ort -d path/to/model_sample.onnx \
                            -bs "relu_1.tmp_0" \
                            -s 10
```

打印出来的文件信息
```
topology index 12, output relu_1.tmp_0, shape (1, 8, 12, 96), dtype float16
topology index 22, output batch_norm_4.tmp_3, shape (1, 24, 12, 96), dtype float16
topology index 32, output depthwise_conv2d_2.tmp_0, shape (1, 32, 6, 96), dtype float16
topology index 42, output p2o.Mul.5, shape (1, 32, 6, 96), dtype float16
topology index 52, output relu_7.tmp_0, shape (1, 8, 1, 1), dtype float16
topology index 54, output hardsigmoid_1.tmp_0, shape (1, 32, 1, 1), dtype float16
```

上述命令中如果不指定 compare_dir & golden_dir 就不会对输出的文件进行比较，不会产生 compare_result 文件，如果仅指定 compare_dir 会将output 与 compare 文件进行比较，且会产生如下数据：1.index 2.file_name 3.nan_inf 4.diff2_value 5.all_close 6.精度阈值, 实例如下：
```
python dump_tools.py dump_once -f sdaa -d ./model.onnx -n relu_3.tmp_0 -cd /paht/to/compare_file

index: 0 | relu_3.tmp_0.npy | nan_inf: False | diff2: 0.0 | all_close: True | th: 1e-06
```

如果 compare_dir & golden_dir文件路径会产生如下数据：
1.index 2.file_name 3.nan_inf 4.diff2_value 5.dif3_max 6.diff3_mean 7.diff3_max_sw 8.diff3_max_g 9.diff3_mean_sw 10.diff3_mean_g, 实例如下：
```
python dump_tools.py dump_once -f sdaa -d ./model.onnx -n relu_3.tmp_0 -cd /paht/to/compare_file -g /paht/to/golden_file

index: 0 | relu_3.tmp_0.npy | nan_inf: False | diff2: 0.0 | diff3_max=pass | diff3_mean=pass | diff3_max_sw: 0.0 | diff3_max_g: 0.0*10.0=0.0 | diff3_mean_sw: 0.0 | diff3_mean_g: 0.0*10.0=0.0
```

如果在定位精度问题时，想查看对应 op 的信息，例如attrs，inputs，outputs等信息时，通过使用 dump_once 指令，将对应的output name传入即可将对应的模型的 relay ir 打印出来。
打印模型中融合算子 conv_bn_act 的relay ir， 实例如下：
```
使用指令 dump_once 指定要打印的 output name： relu_3.tmp_0
python dump_tools.py dump_once -f sdaa -d ./model_sample.onnx -n relu_3.tmp_0

self.ir_module:  def @main(%eager_tmp_0: Tensor[(1, 3, 48, 192), float32] /* ty=Tensor[(1, 3, 48, 192), float32] */) -> Tensor[(1, 24, 12, 96), float16] {
  %0 = cast(%eager_tmp_0, dtype="float16") /* ty=Tensor[(1, 3, 48, 192), float16] */;
  %1 = layout_transform(%0, src_layout="NCHW", dst_layout="NHWC") /* ty=Tensor[(1, 48, 192, 3), float16] */;
  %2 = nn.conv2d_batchnorm(%1, meta[relay.Constant][0] /* ty=Tensor[(3, 3, 3, 8), float16] */, meta[relay.Constant][1] /* ty=Tensor[(8), float16] */, meta[relay.Constant][2] /* ty=Tensor[(8), float16] */, meta[relay.Constant][3] /* ty=Tensor[(8), float16] */, meta[relay.Constant][4] /* ty=Tensor[(8), float16] */, strides=[2, 2], padding=[1, 1, 1, 1], channels=8, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="IHWO", axis=3);
  %3 = add(%2, meta[relay.Constant][5] /* ty=Tensor[(1), float16] */) /* ty=Tensor[(1, 24, 96, 8), float16] */;
  %4 = clip(%3, a_min=0f, a_max=6f) /* ty=Tensor[(1, 24, 96, 8), float16] */;
  %5 = multiply(%2, %4) /* ty=Tensor[(1, 24, 96, 8), float16] */;
  %6 = divide(%5, meta[relay.Constant][6] /* ty=Tensor[(1), float16] */) /* ty=Tensor[(1, 24, 96, 8), float16] */;
  %7 = nn.conv2d_batchnorm_activation(%6, meta[relay.Constant][7] /* ty=Tensor[(8, 1, 1, 8), float16] */, meta[relay.Constant][8] /* ty=Tensor[(8), float16] */, meta[relay.Constant][9] /* ty=Tensor[(8), float16] */, meta[relay.Constant][10] /* ty=Tensor[(8), float16] */, meta[relay.Constant][11] /* ty=Tensor[(8), float16] */, padding=[0, 0, 0, 0], channels=8, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="IHWO", axis=3);
  %8 = nn.conv2d_batchnorm_activation(%7, meta[relay.Constant][12] /* ty=Tensor[(1, 3, 3, 8), float16] */, meta[relay.Constant][13] /* ty=Tensor[(8), float16] */, meta[relay.Constant][14] /* ty=Tensor[(8), float16] */, meta[relay.Constant][15] /* ty=Tensor[(8), float16] */, meta[relay.Constant][16] /* ty=Tensor[(8), float16] */, strides=[2, 1], padding=[1, 1, 1, 1], groups=8, channels=8, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="IHWO", axis=3);
  %9 = nn.global_avg_pool2d(%8, layout="NHWC") /* ty=Tensor[(1, 1, 1, 8), float16] */;
  %10 = nn.conv2d(%9, meta[relay.Constant][17] /* ty=Tensor[(8, 1, 1, 2), float16] */, padding=[0, 0, 0, 0], channels=2, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="IHWO") /* ty=Tensor[(1, 1, 1, 2), float16] */;
  %11 = add(%10, meta[relay.Constant][18] /* ty=Tensor[(1, 1, 1, 2), float16] */) /* ty=Tensor[(1, 1, 1, 2), float16] */;
  %12 = nn.relu(%11) /* ty=Tensor[(1, 1, 1, 2), float16] */;
  %13 = nn.conv2d(%12, meta[relay.Constant][19] /* ty=Tensor[(2, 1, 1, 8), float16] */, padding=[0, 0, 0, 0], channels=8, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="IHWO") /* ty=Tensor[(1, 1, 1, 8), float16] */;
  %14 = add(%13, meta[relay.Constant][20] /* ty=Tensor[(1, 1, 1, 8), float16] */) /* ty=Tensor[(1, 1, 1, 8), float16] */;
  %15 = multiply(%14, 0.199951f16 /* ty=float16 */) /* ty=Tensor[(1, 1, 1, 8), float16] */;
  %16 = add(%15, 0.5f16 /* ty=float16 */) /* ty=Tensor[(1, 1, 1, 8), float16] */;
  %17 = clip(%16, a_min=0f, a_max=1f) /* ty=Tensor[(1, 1, 1, 8), float16] */;
  %18 = multiply(%8, %17) /* ty=Tensor[(1, 12, 96, 8), float16] */;
  %19 = nn.conv2d_batchnorm(%18, meta[relay.Constant][21] /* ty=Tensor[(8, 1, 1, 8), float16] */, meta[relay.Constant][22] /* ty=Tensor[(8), float16] */, meta[relay.Constant][23] /* ty=Tensor[(8), float16] */, meta[relay.Constant][24] /* ty=Tensor[(8), float16] */, meta[relay.Constant][25] /* ty=Tensor[(8), float16] */, padding=[0, 0, 0, 0], channels=8, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="IHWO", axis=3);
  %20 = nn.conv2d_batchnorm_activation(%19, meta[relay.Constant][26] /* ty=Tensor[(8, 1, 1, 24), float16] */, meta[relay.Constant][27] /* ty=Tensor[(24), float16] */, meta[relay.Constant][28] /* ty=Tensor[(24), float16] */, meta[relay.Constant][29] /* ty=Tensor[(24), float16] */, meta[relay.Constant][30] /* ty=Tensor[(24), float16] */, padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1], data_layout="NHWC", kernel_layout="IHWO", axis=3);
  layout_transform(%20, src_layout="NHWC", dst_layout="NCHW") /* ty=Tensor[(1, 24, 12, 96), float16] */
}

 打印中的最后一层为对应的 conv_bn_act 的 relay 信息，通过该 relay ir 可以查看到对应的attrs, layerout 等上下文信息，可以使用该信息构建对应的测试用例。

```


# dump工具介绍

## 一、概述

当模型推理出现精度错误时，通常需要排查是模型中的哪一个算子导致的问题，这时可以通过dump工具在固定输入的前提下输出模型中间层的输出，并和参考输出比较确定错误位置；

## 二、方案设计

目前采用的实现方案是：通过反复截取原模型，得到以某个tensor为输出的子模型，并对子模型进行推理获得对应tensor的输出结果；

流程如下：

1. 分析onnx模型，获取模型的输入tensor名称、每个节点的输出tensor名称（包含模型的输出tensor），根据onnx模型的规则，获得每个节点输出tensor的顺序是符合拓扑排序的；
2. 固定模型的输入名称，按拓扑序依次遍历输出节点tensor，截取只包含单个输出的子模型，例如原模型的节点为（0，1，2，3，4，5），其中0为输入，5为输出，中间为内部节点tensor，则截取的子模型分别为（0，1）、（0，1，2）、（0，1，2，3）、（0，1，2，3，4）、（0，1，2，3，4，5）；
3. 固定输入内容，将每个子模型通过推理计算得到输出，并记录为对应节点tensor的输出；

目的：

1. 保证推理时中间结点的输出结果格式与模型是一致的，因为没有标记成输出的节点tensor可能会使用其他的格式用于加速（NHWC、NCHWc4等），标记后可以通过添加额外的格式转换和模型对齐；
2. 在遇到多个算子融合的情况时，至少会有一个子模型包含了融合需要的全部算子；

缺点：

1. 模型截取和模型运行的次数可能会非常多，导致整体耗时过长；
2. 暂时没有考虑到单个节点存在多个输出的情况是否能覆盖；

## 三、实现

考虑到使用该工具的场景多为开发过程中，需要经常对程序修改，因此目前的实现采用对外提供多种功能接口的方式，允许开发人员自行组合调用；

### 1. executor

executor是对推理后端的抽象，包括四个接口：init，load_model，execute，release；实际使用时会按照该顺序依次调用接口，完成一次推理流程；

对于一个具体的推理后端，需要继承BaseExecutor这个类并实现接口，如果没有对应的实现可以置空；

### 2. dump_info

dump_info主要用于在运行过程中添加和保存相关信息，比如模型信息、输入tensor信息、输出tensor信息，对于tensor还会保存为npy文件；

dump的内容包括一个json描述文件，输入和输出的tensor会分别保存在不同的目录下（除非设置为同一个），用于后续的精度比较；

### 3. pipeline

pipeline对dump流程进行封装，初始化阶段可以传入模型信息、配置信息、dump_info，另外可以对模型截取或标记输出，设置输入数据后执行dump功能，将结果存储到指定目录下；

## 四、示例

目前提供了几个示例代码，可直接运行 `python xxx.py` ，

### 1. sample0.py

最基本的用法，使用onnxruntime后端计算；
示例中提供了3种输入模式，分别是使用默认随机范围生成、使用指定随机范围生成、使用外部数据输入；

### 2. sample1.py

示例中提供了2种特殊输入模式，分别是从某个生成的json读取随机范围生成、从某个生成的npy文件读取array数据；

### 3. sample2.py

示例中提供了截取模型的方式，执行extract_subgraph后pipeline内部的模型就切换为截取后的模型；

### 4. sample3.py

示例为标记模型输出和单次运行；

dump_once和dump_full的区别是dump_once只按照当前的模型（原始或截取后）运行一次并拿到模型输出，dump_full是对mark_outputs和dump_once的简单封装，具体是按拓扑顺序遍历当前模型的节点输出，每次只标记一个输出，然后dump_once获得结果，最终运行若干次后相当于获得所有节点输出的结果；

### 5. sample4.py

使用sdaa tvm作为executor的示例，调用过程和onnxruntime的executor相同，原则上不同的executor可以相互替换；

### 6. sample5.py

一些内部工具的使用示例，可根据需要调用

### 7. sample6.py

dump_list提供了自定义输出列表的方法，使用者可以指定需要dump的tensor名称，每一次运行中程序会标记列表中的一个名称，并通过dump_once获得结果;

### 8. sample7.py

dump_range提供了自定义输出范围和和步长的方法，使用者可以指定输出的开始和结束，可以使用名称或拓扑顺序的索引值，设置步长值可以选择相邻输出的间隔，可以用于减少计算次数，无论步长值设置为多大，开始和结束的tensor一定会输出;

## Tips

1. 同一个模型尽量使用单batch的版本，可减少计算时间，保存的文件也相对小一些；
2. onnxruntime可以作为参考输出，一般同一个模型只需要获得一次onnxruntime的输出，后续侧重于调试其他后端；
3. 如果某个模型较大，层数较多，可以先观察结构，人为将模型分成几个部分，分别使用dump输出，减少加载模型的时间；

## 后续

如果sdaa tvm的接口可以统一，就可以像onnxruntime一样封装为一个通用的executor，dump工具可以做成类似命令行的工具；
