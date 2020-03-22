import torch
CUDA = torch.cuda.is_available()
if CUDA:
    print('yes')
else:
    print('no')