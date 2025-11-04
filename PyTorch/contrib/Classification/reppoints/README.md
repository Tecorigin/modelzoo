# RepPoints

## 1. 模型概述
RepPoints 是一种基于表示点（representation points）进行目标检测的方法，它提出了一种新的思路来替代传统的框回归（bounding box regression）方法，旨在通过更灵活的点集来表示物体的形状和位置。这一方法由 RepPoints: Confidence-Driven Object Detection with Recurrent Point Generation 论文中提出，主要目标是提升目标检测的精度和鲁棒性，尤其是在复杂场景中。

源码链接: https://github.com/open-mmlab/mmdetection

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

#### 2.2.1 数据集位置
/data/datasets/20241122/coco/


2. 安装python依赖
``` 
bash
# install requirements
pip install -r requirements.txt
```
### 2.4 启动训练
1. 在训练之前，我们需要对mmengine库进行必要的修改，这里的路径中mjytorch为本人的环境名，需要更改。

首先安装需要的库：
``` 
pip install -r requirements.txt
```
接着请开始修改：
#### 1.修改了/root/miniconda3/envs/mjytorch/lib/python3.10/site-packages/mmengine/optim/optimizer/amp_optimizer_wrapper.py
第76-77行:
``` 
assert is_cuda_available() or is_npu_available() or is_mlu_available(
        ) or is_musa_available() or torch.sdaa.is_available(), (
```
#### 2.修改了/root/anaconda3/envs/mjytorch/lib/python3.10/site-packages/mmengine/device/utils.py
第53-58行：
```
mem = torch.sdaa.max_memory_allocated(device=device)
mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int,
                          device=device)
torch.sdaa.reset_peak_memory_stats()
return int(mem_mb.item())
```
第63行:
 ```
 return torch.sdaa.is_available()
 ```
#### 3.修改了/root/miniconda3/envs/mjytorch/lib/python3.10/site-packages/mmengine/dist/utils.py
第130行：
```
torch.sdaa.set_device(local_rank)
```
第133行：
```
torch_dist.init_process_group(backend='tccl', **kwargs)
```
#### 4.修改了/root/anaconda3/envs/mjytorch/lib/python3.10/site-packages/mmengine/runner/runner.py
将877行的model = model.to(get_device()) 改为model = model.to("sdaa")
第2042行的device = get_device() 改为 device = "sdaa"

#### 5./root/miniconda3/envs/mjytorch/lib/python3.10/site-packages/mmcv/ops/roi_align.py

第91行加入input_cloned = input.clone()

将下一行ext_module.roi_align_forward中的input改为input_cloned

#### 6. /root/miniconda3/envs/mjytorch/lib/python3.10/site-packages/mmengine/structures/instance_data.py

25-26行改为：（cuda改为sdaa）
```
else:
    BoolTypeTensor = Union[torch.BoolTensor, torch.sdaa.BoolTensor]
    LongTypeTensor = Union[torch.LongTensor, torch.sdaa.LongTensor]
```
#### 7./root/miniconda3/envs/mjytorch/lib/python3.10/site-packages/mmcv/ops/deform_conv.py
class DeformConv2dFunction(Function)中的forward与backward两个函数修改为：
```
    def forward(ctx,
                input: Tensor,
                offset: Tensor,
                weight: Tensor,
                stride: Union[int, Tuple[int, ...]] = 1,
                padding: Union[int, Tuple[int, ...]] = 0,
                dilation: Union[int, Tuple[int, ...]] = 1,
                groups: int = 1,
                deform_groups: int = 1,
                bias: bool = False,
                im2col_step: int = 32) -> Tensor:
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        assert bias is False, 'Only support bias is False.'
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.im2col_step = im2col_step
        ctx.device = input.device.type

            # Ensure input, offset, and weight are all on CPU
        # if input.device.type != 'cpu':
        #     input = input.cpu()
        # if offset.device.type != 'cpu':
        #     offset = offset.cpu()
        # if weight.device.type != 'cpu':
        #     weight = weight.cpu()
        
        # Make input contiguous
        # input = input.contiguous()
        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        weight = weight.type_as(input)
        if ctx.device == 'npu':
            mask_shape, _ = torch.chunk(offset, 2, dim=1)
            mask = torch.ones_like(mask_shape).to(input.device)
            bias = input.new_empty(0)
            output = ModulatedDeformConv2dFunction._npu_forward(
                ctx, input, offset, mask, weight, bias)
            return output
        
        input = input.contiguous()  # 确保 input 张量是连续的
        offset = offset.contiguous()  # 确保 offset 张量是连续的
        weight = weight.contiguous()  # 确保 weight 张量是连续的
        

        device = input.device  # 保存原来的设备
        input_cpu = input.cpu()  # 将 input 转移到 CPU
        offset_cpu = offset.cpu()  # 将 offset 转移到 CPU
        weight_cpu = weight.cpu()  # 将 weight 转移到 CPU
        
        #bias_cpu = bias.cpu()
        ctx.save_for_backward(input, offset, weight)
        
        output = input.new_empty([
            int(i)
            for i in DeformConv2dFunction._output_size(ctx, input, weight)
        ])
        output = output.contiguous()
        output = output.cpu()

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        assert (input.size(0) % cur_im2col_step
                ) == 0, 'batch size must be divisible by im2col_step'
        

        ext_module.deform_conv_forward(
            input_cpu,
            weight_cpu,
            offset_cpu,
            output,
            ctx.bufs_[0].cpu(),
            ctx.bufs_[1].cpu(),
            kW=weight.size(3),
            kH=weight.size(2),
            dW=ctx.stride[1],
            dH=ctx.stride[0],
            padW=ctx.padding[1],
            padH=ctx.padding[0],
            dilationW=ctx.dilation[1],
            dilationH=ctx.dilation[0],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            im2col_step=cur_im2col_step)
        # input = input.to('sdaa',non_blocking=True)
        # output = output.to('sdaa',non_blocking=True)
        # offset = offset.to('sdaa',non_blocking=True)
        # weight = weight.to('sdaa',non_blocking=True)

        return output.to(device)

    @staticmethod
    @once_differentiable
    def backward(
        ctx, grad_output: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None,
               None, None, None, None, None, None]:
        if ctx.device == 'npu':
            return DeformConv2dFunction._npu_backward(ctx, grad_output)
        input, offset, weight = ctx.saved_tensors

        device = input.device  # 保存原来的设备
        input_cpu = input.cpu()  # 将 input 转移到 CPU
        offset_cpu = offset.cpu()  # 将 offset 转移到 CPU
        weight_cpu = weight.cpu()  # 将 weight 转移到 CPU

        grad_input = grad_offset = grad_weight = None

        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        assert (input.size(0) % cur_im2col_step
                ) == 0, 'batch size must be divisible by im2col_step'

        grad_output = grad_output.contiguous()
        grad_output = grad_output.cpu()
        

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_input = torch.zeros_like(input)
            grad_offset = torch.zeros_like(offset)
            grad_input = grad_input.contiguous()
            grad_input = grad_input.cpu()
            grad_offset = grad_offset.contiguous()
            grad_offset = grad_offset.cpu()
            ext_module.deform_conv_backward_input(
                input_cpu,
                offset_cpu,
                grad_output,
                grad_input,
                grad_offset,
                weight_cpu,
                ctx.bufs_[0].cpu(),
                kW=weight.size(3),
                kH=weight.size(2),
                dW=ctx.stride[1],
                dH=ctx.stride[0],
                padW=ctx.padding[1],
                padH=ctx.padding[0],
                dilationW=ctx.dilation[1],
                dilationH=ctx.dilation[0],
                group=ctx.groups,
                deformable_group=ctx.deform_groups,
                im2col_step=cur_im2col_step)

        if ctx.needs_input_grad[2]:
            grad_weight = torch.zeros_like(weight)
            grad_weight = grad_weight.contiguous()
            grad_weight = grad_weight.cpu()

            ext_module.deform_conv_backward_parameters(
                input_cpu,
                offset_cpu,
                grad_output,
                grad_weight,
                ctx.bufs_[0].cpu(),
                ctx.bufs_[1].cpu(),
                kW=weight.size(3),
                kH=weight.size(2),
                dW=ctx.stride[1],
                dH=ctx.stride[0],
                padW=ctx.padding[1],
                padH=ctx.padding[0],
                dilationW=ctx.dilation[1],
                dilationH=ctx.dilation[0],
                group=ctx.groups,
                deformable_group=ctx.deform_groups,
                scale=1,
                im2col_step=cur_im2col_step)

        return grad_input.to(device), grad_offset.to(device), grad_weight.to(device), \
            None, None, None, None, None, None, None
```
2. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/reppoints
    ```

- 单机单核组
    ```
    torchrun tools/train.py configs/reppoints/reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco.py
    ```
- 单机单卡
    ```
    torchrun tools/train.py configs/reppoints/reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco.py --launcher pytorch --amp
    ```


### 2.5 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| RepPoints |是|3|300*300|



