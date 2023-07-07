import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import ToPILImage
from IPython.display import display
import math


#######################################################################
#                          Forward Process                            #
#######################################################################

import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader


# beta scheduler
def linear_beta_schedule(timesteps, start = 0.0001, end=0.02):
    return torch.linspace(start, end, timesteps) 

def cosine_beta_schedule(timesteps, start = 0.0001, end = 0.02):
    betas = torch.linspace(0, 1, timesteps)
    betas = 0.5 * (1 + torch.cos(betas * math.pi))
    betas = start + (end - start) * betas
    return betas

'''
특정 timestep t에서의 값을 가져오기 위한 장치
vals: timestep에 다라 변하는 값을 가진 리스트 (2차원)
t: 특정 timestep의 시간을 나타내는 tensor (2차원)
'''
def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0] # t는 embedding된 timestep의 tensor
    out = vals.gather(-1, t.cpu())  # t에 해당하는 값을 가져옴.
    
    return out.reshape(batch_size, *((1, ) * (len(x_shape) - 1))).to(t.device) # out tensor의 형태를 조정. (batch_size, 1,1,1,...,1) 로 바꿔서 x_0와 동일한 형태


# x_0와 timestep T를 input으로 받고 noise로 만들기
def forward_diffusion_sample(x_0, t, device="cpu"):
    
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

T = 1000 # timestep T
betas = linear_beta_schedule(timesteps=T)

alphas = 1 - betas  
alphas_cumprod = torch.cumprod(alphas, axis=0) # noise 크기의 누적 변화
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1) # 이전 timestep 값
sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # 1 / alpha = noise의 variance 
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # alphas_cumprod: noise 크기의 누적 변화 (\Pi alpha)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod) 
posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

########################################################################
#                          Denoising process                           #
########################################################################

def reverse_process(x_T, x_0):
    x_t = x_T

    for idx in range(T-1, -1, -1):
        t = torch.Tensor([idx]).type(torch.int64)
        sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) # t에 해당하는 값 가져오기. 노이즈 크기 누적변화
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape) # 노이즈 크기의 변화
        noise = (x_t - sqrt_alphas_cumprod_t * x_0) / sqrt_one_minus_alphas_cumprod_t # 현재 timestep에서 denoising 된 값
        x_t = noise

    return x_t

