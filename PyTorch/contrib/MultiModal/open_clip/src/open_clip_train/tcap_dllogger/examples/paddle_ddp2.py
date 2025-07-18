import random
from typing import Union
from argparse import ArgumentParser

import paddle
import paddle.distributed as dist

# from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_tensor_loss(loss):
    if not is_dist_avail_and_initialized():
        return
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with paddle.no_grad():
        dist.all_reduce(loss,op=dist.ReduceOp.SUM)
       
        loss_mean = loss / world_size
    return loss_mean


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with paddle.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = paddle.stack(values)
        dist.all_reduce(values,op=dist.ReduceOp.SUM)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def reduce_metric(input:Union[paddle.Tensor, dict], is_dict=False, mode='mean'):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    reduce_mode = {
        "sum":dist.ReduceOp.SUM,
        "min":dist.ReduceOp.MIN,
        "max":dist.ReduceOp.MAX,
        "prod":dist.ReduceOp.PROD,
        "mean":dist.ReduceOp.SUM,
    }
    assert mode in ['sum', 'mean','min', 'max','prod'], \
        f"Value error, reduce ops support only ['sum', 'mean','min', 'max','prod'], but got {mode}." 
    world_size = get_world_size()
    if world_size < 2:
        return input
    
    if is_dict:
        with paddle.no_grad():
            names = []
            values = []
            # sort the keys so that they are consistent across processes
            for k in sorted(input.keys()):
                names.append(k)
                values.append(input[k])
            values = paddle.stack(values)
            dist.all_reduce(values,op=reduce_mode[mode])
            if mode == "mean":
                values /= world_size
            reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict
    else:
        with paddle.no_grad():
            dist.all_reduce(loss,op=reduce_mode[mode])
            if mode == "mean":
                loss_mean = loss / world_size
        return loss_mean



if __name__ == "__main__":
    parser = ArgumentParser(prog="tcaplogger for paddle ddp.")
    parser.add_argument("--local_rank", default=-1)
    parser.add_argument("--device", default='swai')
    args = parser.parse_args()
    
    local_rank = args.local_rank
    device = args.device
    paddle.set_device(device)

    dist.init_parallel_env()

    for idx in range(3):
        if is_main_process():
            loss = paddle.to_tensor(7.0)
        else:
            loss = paddle.to_tensor(1.0)
        speed = paddle.to_tensor(float(random.random())*10)
        dict_loss = {"loss_total":loss}
        dict_speed = {"speed_total":speed}
        # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        # reduced_loss = reduce_tensor_loss(loss)
        reduced_dict = reduce_metric(dict_loss,is_dict=True)

        if is_main_process():
            # print(">>>>>>> ",reduced_loss.item())
            print(">>>>>>> ", reduced_dict['loss_total'].item()) # 
