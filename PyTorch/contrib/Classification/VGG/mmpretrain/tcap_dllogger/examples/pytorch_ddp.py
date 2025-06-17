import random
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

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
    with torch.no_grad():
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
    print(input_dict)
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values,op=dist.ReduceOp.SUM)
        if average:
            values /= world_size
        print(values)
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


if __name__ == "__main__":
    parser = ArgumentParser(prog="tcaplogger for pytorch ddp.")
    parser.add_argument("--local_rank", default=-1)
    parser.add_argument("--device", default='swai')
    args = parser.parse_args()
    if args.local_rank != -1:
        local_rank = 0
        if 'cuda' in args.device:
            local_rank  = args.local_rank
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        elif 'swai' in args.device:
            import torch_swai.core.swai_model as sm
            local_rank  = args.local_rank
            sm.set_device(int(local_rank))
            dist.init_process_group(backend="tccl")
        else:
            raise ValueError("only support DDP for SWAI or GPU device.")

    json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, 'tmp.json'),
    ]
    )
    print("11111111111")
    json_logger.metadata("loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata(
        "speed",
        {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"},
    )

    for idx in range(3):
        
        loss = torch.tensor(float(local_rank)).to('swai')
        speed = torch.tensor(float(local_rank)*10).to('swai')
        dict_loss = {"loss_total":loss}
        dict_speed = {"speed_total":speed}
        

        # reduced_loss = reduce_tensor_loss(loss)
        # reduced_speed = reduce_tensor_loss(speed)

        print(dict_loss)
        reduced_loss_dict = reduce_dict(dict_loss)
        # reduced_speed_dict = reduce_dict(dict_speed)

        # json_logger.log(
        #     step = (idx,),
        #     data = {"loss":str(loss.item()), 
        #             "speed":str(speed.item())},
        #     verbosity=Verbosity.DEFAULT,
        # )
        if is_main_process():
        #     json_logger.log(
        #     step = (idx,),
        #     data = {"loss":str(reduced_loss.item()), 
        #             "speed":str(reduced_speed.item())},
        #     verbosity=Verbosity.DEFAULT,
        # )
            json_logger.log(
            step = (idx,),
            data = {"dict_loss":str(reduced_loss_dict['loss_total'].item()), 
                    # "dict_speed":str(reduced_speed['speed_total'].item())
                    },
            verbosity=Verbosity.DEFAULT,
        )
