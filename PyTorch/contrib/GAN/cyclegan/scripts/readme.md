
记载：/root/miniconda3/envs/cyclegan/lib/python3.10/site-packages/torch/nn/modules/padding.py，将F.pad放到cpu上运算将_ReflectionPadNd：
class _ReflectionPadNd(Module):
    __constants__ = ['padding']
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        '''print("======")
        print(input.shape)
        print(self.padding)'''
        input_copy = input
        input_copy = input_copy.cpu()
        output = F.pad(input_copy, self.padding, 'reflect')
        
        output_copy = output
        
        return output_copy.to("sdaa")

改为：

import torch
def aligned_cpu_tensor(shape, dtype=torch.float32, alignment=64):
    numel = torch.Size(shape).numel()
    element_size = torch.tensor([], dtype=dtype).element_size()
    bytes_need = numel * element_size
    bytes_need = (bytes_need + alignment - 1) // alignment * alignment  # 对齐到alignment字节
    
    # 分配对齐内存并截断到精确长度
    storage = torch.ByteStorage.from_buffer(bytearray(bytes_need), 'cpu')
    tensor = torch.tensor(storage, dtype=torch.uint8)
    return tensor[:numel * element_size].view(dtype).view(shape)

class _ReflectionPadNd(Module):
    __constants__ = ['padding']
    padding: Sequence[int]

    def forward(self, x):
        input_cpu = x.cpu()
        h_pad = self.padding[2] + self.padding[3]
        w_pad = self.padding[0] + self.padding[1]
        output_shape = (
            input_cpu.size(0),
            input_cpu.size(1),
            input_cpu.size(2) + h_pad,
            input_cpu.size(3) + w_pad
        )
        output_cpu = aligned_cpu_tensor(output_shape, dtype=input_cpu.dtype)
        F.pad(input_cpu, self.padding, 'reflect')
        return output_cpu.contiguous().to(x.device)
    def extra_repr(self) -> str:
        return f'{self.padding}'