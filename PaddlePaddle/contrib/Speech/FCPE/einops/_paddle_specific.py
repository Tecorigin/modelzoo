"""
Specialization of einops for torch.

Unfortunately, torch's jit scripting mechanism isn't strong enough,
and to have scripting supported at least for layers,
a number of changes is required, and this layer helps.

Importantly, whole lib is designed so that you can't use it
"""

from typing import Dict, List

import paddle
from einops.einops import TransformRecipe, _reconstruct_from_shape_uncached


class PaddleJitBackend:
    """
    Completely static backend that mimics part of normal backend functionality
    but restricted to torch stuff only
    """

    @staticmethod
    def reduce(x: paddle.Tensor, operation: str, reduced_axes: List[int]):
        if operation == 'min':
            return x.amin(axis=reduced_axes)
        elif operation == 'max':
            return x.amax(axis=reduced_axes)
        elif operation == 'sum':
            return x.sum(axis=reduced_axes)
        elif operation == 'mean':
            return x.mean(axis=reduced_axes)
        elif operation == 'prod':
            for i in list(sorted(reduced_axes))[::-1]:
                x = x.prod(axis=i)
            return x
        else:
            raise NotImplementedError('Unknown reduction ', operation)

    @staticmethod
    def transpose(x, axes: List[int]):
        return x.permute(axes)

    @staticmethod
    def stack_on_zeroth_dimension(tensors: List[paddle.Tensor]):
        return paddle.stack(tensors)

    @staticmethod
    def tile(x, repeats: List[int]):
        return x.repeat(repeats)

    @staticmethod
    def add_axes(x, n_axes: int, pos2len: Dict[int, int]):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = paddle.unsqueeze(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    @staticmethod
    def is_float_type(x):
        return x.dtype in [paddle.float16, paddle.float32, paddle.float64, paddle.bfloat16]

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def reshape(x, shape: List[int]):
        return x.reshape(shape)


# mirrors einops.einops._apply_recipe
def apply_for_scriptable_paddle(recipe: TransformRecipe, tensor: paddle.Tensor, reduction_type: str) -> paddle.Tensor:
    backend = PaddleJitBackend
    init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes = \
        _reconstruct_from_shape_uncached(recipe, backend.shape(tensor))
    tensor = backend.reshape(tensor, init_shapes)
    if len(reduced_axes) > 0:
        tensor = backend.reduce(tensor, operation=reduction_type, reduced_axes=reduced_axes)
    tensor = backend.transpose(tensor, axes_reordering)
    if len(added_axes) > 0:
        tensor = backend.add_axes(tensor, n_axes=len(axes_reordering) + len(added_axes), pos2len=added_axes)
    return backend.reshape(tensor, final_shapes)
