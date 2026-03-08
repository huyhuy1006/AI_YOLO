import torch
checkpoint = torch.load('outputs/checkpoints/last.pt')
print(checkpoint.keys())