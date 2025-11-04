import torch

query = torch.load('query.pt')
key = torch.load('key.pt')

device = 'sdaa'
query = query.to(device)
key = key.to(device)
breakpoint()
res = torch.matmul(query,key)
