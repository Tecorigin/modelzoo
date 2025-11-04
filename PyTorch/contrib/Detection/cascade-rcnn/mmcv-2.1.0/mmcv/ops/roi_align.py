# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
import torch_sdaa
import torch.nn as nn
from mmengine.utils import deprecated_api_warning
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['roi_align_forward', 'roi_align_backward'])


class RoIAlignFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale, sampling_ratio,
                 pool_mode, aligned):
        from torch.onnx import TensorProtoDataType
        from torch.onnx.symbolic_opset9 import sub

        def _select(g, self, dim, index):
            return g.op('Gather', self, index, axis_i=dim)

        # batch_indices = rois[:, 0].long()
        batch_indices = _select(
            g, rois, 1,
            g.op('Constant', value_t=torch.tensor([0], dtype=torch.long)))
        batch_indices = g.op('Squeeze', batch_indices, axes_i=[1])
        batch_indices = g.op(
            'Cast', batch_indices, to_i=TensorProtoDataType.INT64)
        # rois = rois[:, 1:]
        rois = _select(
            g, rois, 1,
            g.op(
                'Constant',
                value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))

        if aligned:
            # rois -= 0.5/spatial_scale
            aligned_offset = g.op(
                'Constant',
                value_t=torch.tensor([0.5 / spatial_scale],
                                     dtype=torch.float32))
            rois = sub(g, rois, aligned_offset)
        # roi align
        return g.op(
            'RoiAlign',
            input,
            rois,
            batch_indices,
            output_height_i=output_size[0],
            output_width_i=output_size[1],
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=max(0, sampling_ratio),
            mode_s=pool_mode)

    @staticmethod
    def forward(ctx: Any,
                input: torch.Tensor,
                rois: torch.Tensor,
                output_size: int,
                spatial_scale: float = 1.0,
                sampling_ratio: int = 0,
                pool_mode: str = 'avg',
                aligned: bool = True) -> torch.Tensor:
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        assert pool_mode in ('max', 'avg')
        ctx.pool_mode = 0 if pool_mode == 'max' else 1
        ctx.aligned = aligned
        ctx.input_shape = input.size()

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)
        if ctx.pool_mode == 0:
            argmax_y = input.new_zeros(output_shape)
            argmax_x = input.new_zeros(output_shape)
        else:
            argmax_y = input.new_zeros(0)
            argmax_x = input.new_zeros(0)

        input = input.contiguous()

        ext_module.roi_align_forward(
        # roi_align_forward(
            input,
            rois,
            output,
            argmax_y,
            argmax_x,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)
        
        ctx.save_for_backward(rois, argmax_y, argmax_x)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        rois, argmax_y, argmax_x = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        # complex head architecture may cause grad_output uncontiguous.
        grad_output = grad_output.contiguous()
        ext_module.roi_align_backward(
            grad_output,
            rois,
            argmax_y,
            argmax_x,
            grad_input,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)
        return grad_input, None, None, None, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):
    """RoI align pooling layer.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
        use_torchvision (bool): whether to use roi_align from torchvision.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/

        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;

        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """

    @deprecated_api_warning(
        {
            'out_size': 'output_size',
            'sample_num': 'sampling_ratio'
        },
        cls_name='RoIAlign')
    def __init__(self,
                 output_size: tuple,
                 spatial_scale: float = 1.0,
                 sampling_ratio: int = 0,
                 pool_mode: str = 'avg',
                 aligned: bool = True,
                 use_torchvision: bool = False):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned
        self.use_torchvision = use_torchvision

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
        """
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            if 'aligned' in tv_roi_align.__code__.co_varnames:
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio,
                                    self.aligned)
            else:
                if self.aligned:
                    rois -= rois.new_tensor([0.] +
                                            [0.5 / self.spatial_scale] * 4)
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio)
        else:
            return roi_align(input, rois, self.output_size, self.spatial_scale,
                             self.sampling_ratio, self.pool_mode, self.aligned)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'pool_mode={self.pool_mode}, '
        s += f'aligned={self.aligned}, '
        s += f'use_torchvision={self.use_torchvision})'
        return s
    
    
def pre_calc_for_bilinear_interpolate(height, width, pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w, roi_start_h, roi_start_w, bin_size_h, bin_size_w):
    pre_calc = []
    for ph in range(pooled_height):
        for pw in range(pooled_width):
            for iy in range(roi_bin_grid_h):
                yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                for ix in range(roi_bin_grid_w):
                    xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w

                    x = xx
                    y = yy

                    if y < -1.0 or y > height or x < -1.0 or x > width:
                        pre_calc.append((0, 0, 0, 0, 0, 0, 0, 0))
                        continue

                    if y <= 0:
                        y = 0
                    if x <= 0:
                        x = 0

                    y_low = int(y)
                    x_low = int(x)
                    if y_low >= height - 1:
                        y_high = y_low = height - 1
                        y = y_low
                    else:
                        y_high = y_low + 1

                    if x_low >= width - 1:
                        x_high = x_low = width - 1
                        x = x_low
                    else:
                        x_high = x_low + 1

                    ly = y - y_low
                    lx = x - x_low
                    hy = 1. - ly
                    hx = 1. - lx

                    w1 = hy * hx
                    w2 = hy * lx
                    w3 = ly * hx
                    w4 = ly * lx

                    pre_calc.append((y_low * width + x_low, y_low * width + x_high, y_high * width + x_low, y_high * width + x_high, w1, w2, w3, w4))

    return pre_calc

def roi_align_forward(input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned):
    channels, height, width = input.shape[1], input.shape[2], input.shape[3]
    n_rois = rois.size(0)
    # 4967424 // 256 // 7 // 7

    output_flat = output.flatten()
    argmax_y_flat = argmax_y.flatten()
    argmax_x_flat = argmax_x.flatten()
    
    pooled_height = aligned_height
    pooled_width = aligned_width
    
    for n in range(n_rois):
        index_n = n * channels * pooled_width * pooled_height
        offset_rois = rois[n]
        roi_batch_ind = int(offset_rois[0])

        offset = 0.5 if aligned else 0.0
        roi_start_w = offset_rois[1] * spatial_scale - offset
        roi_start_h = offset_rois[2] * spatial_scale - offset
        roi_end_w = offset_rois[3] * spatial_scale - offset
        roi_end_h = offset_rois[4] * spatial_scale - offset

        roi_width = max(roi_end_w - roi_start_w, 1.0)
        roi_height = max(roi_end_h - roi_start_h, 1.0)
        if aligned:
            assert roi_width >= 0 and roi_height >= 0, "ROIs in ROIAlign cannot have non-negative size!"

        bin_size_h = roi_height / pooled_height
        bin_size_w = roi_width / pooled_width

        roi_bin_grid_h = sampling_ratio if sampling_ratio > 0 else int(torch.ceil(roi_height / pooled_height))
        roi_bin_grid_w = sampling_ratio if sampling_ratio > 0 else int(torch.ceil(roi_width / pooled_width))
        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)

        pre_calc = pre_calc_for_bilinear_interpolate(height, width, pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w, roi_start_h, roi_start_w, bin_size_h, bin_size_w)

        for c in range(channels):
            index_n_c = index_n + c * pooled_width * pooled_height
            offset_input = input[roi_batch_ind, c].flatten()
            pre_calc_index = 0

            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    index = index_n_c + ph * pooled_width + pw
                    # index = ph * pooled_width + pw

                    output_val = 0.
                    maxval = -float('inf')
                    maxidx_y = -1.
                    maxidx_x = -1.
                    for iy in range(roi_bin_grid_h):
                        for ix in range(roi_bin_grid_w):
                            pc = pre_calc[pre_calc_index]
                            val = pc[4] * offset_input[pc[0]] + pc[5] * offset_input[pc[1]] + pc[6] * offset_input[pc[2]] + pc[7] * offset_input[pc[3]]
                            a = offset_input[pc[0]]
                            if val > maxval:
                                maxval = val
                                maxidx_y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                                maxidx_x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
                            output_val += val
                            pre_calc_index += 1

                    if pool_mode == 0:
                        output_flat[index] = maxval
                        argmax_y_flat[index] = maxidx_y
                        argmax_x_flat[index] = maxidx_x
                    elif pool_mode == 1:
                        output_flat[index] = output_val / count

    output = output_flat.reshape((n_rois, channels, pooled_height, pooled_width))
    argmax_y = argmax_y_flat.reshape((n_rois, channels, pooled_height, pooled_width))
    argmax_x = argmax_x_flat.reshape((n_rois, channels, pooled_height, pooled_width))

    return output, argmax_y, argmax_x