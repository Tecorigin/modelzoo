import torch_sdaa
import torch
print(f"Current device: {torch.sdaa.current_device()}")
torch.sdaa.set_device(5)
print(f"Current device: {torch.sdaa.current_device()}")