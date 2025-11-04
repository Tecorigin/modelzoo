import argparse

import tvm
import tvm.relay

import numpy as np

import time


def parse_args():
    parser = argparse.ArgumentParser(description="the script of executor_run")
    parser.add_argument(
        "--loadEngine", type=str, help="Load a serialized tecoInference engine. "
    )
    parser.add_argument("--input_names", type=str, help="")
    parser.add_argument(
        "--input_shapes",
        type=str,
        help="Set input shapes. Example input shapes spec: --input_shapes=1x3x256x256,1x3x128x128. Each input shape is supplied as a value where value is the dimensions (including the batch dimension) to be used for that input. Multiple input shapes can be provided via comma-separated value.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        default="float16",
        help="Set input shapes dtype. Support type: fp16, fp32, int64. Example input shapes spec: --input_dtype=fp16. Each input shape dtype. If the dtype is set, number must be the same as the input_shapes number. Multiple input shapes dytpe can be provided via comma-separated value(default = float16).",
    )
    parser.add_argument(
        "--warmUp",
        type=int,
        default=10,
        help="Run for N inference iterations to warmup before measure performance (default = 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Run at least N inference iterations (default = 50)",
    )

    return parser.parse_args()


def parse_input_names(name_str):
    return name_str.split(",")


def parse_input_shapes(shape_str):
    shapes_str_list = shape_str.split(",")
    shapes = []
    for it in shapes_str_list:
        dim_str = it.split("x")
        dims = [int(dim) for dim in dim_str]
        shapes.append(dims)

    return shapes


def parse_input_dtype(dtype_str):
    return dtype_str.split(",")


def run(args):
    engine_path = args.loadEngine
    input_names = parse_input_names(args.input_names)
    input_shapes = parse_input_shapes(args.input_shapes)
    input_dtype = parse_input_dtype(args.input_dtype)

    if len(input_dtype) == 1:
        input_dtype *= len(input_shapes)

    assert len(input_shapes) == len(input_names)
    assert len(input_shapes) == len(input_dtype)

    warmUp = args.warmUp
    iterations = args.iterations

    engine = tvm.runtime.create_engine(engine_path, "sdaa")
    ctx = engine.create_context()

    for name, shape, dtype in zip(input_names, input_shapes, input_dtype):
        data = np.random.randn(*shape).astype(dtype)
        ctx.set_input(name, data, tvm.device("sdaa"))

    dev = tvm.device("sdaa")
    for _ in range(warmUp):
        ctx.executor_run(dev=dev)
    dev.sync()

    s = time.time()
    for _ in range(iterations):
        ctx.executor_run(dev=dev)
    dev.sync()
    e = time.time()

    print("executor_run = {} ms".format((e - s) * 1000 / iterations))

    ctx.release()
    engine.release()


if __name__ == "__main__":
    args = parse_args()
    run(args=args)
