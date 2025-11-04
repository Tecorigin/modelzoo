from torch.nn import functional as F
import torch
import torch_sdaa
input_ = torch.randn(2, 256, 64, 64)
padding = (1,1,1,1)
out = F.pad(input_, padding, 'reflect')
out_copy = out.to("sdaa")
print(out_copy)