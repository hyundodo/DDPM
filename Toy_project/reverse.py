import math
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, Dataloader

from einops import rearrange, reduce
# helper functions

# sinusodial positional Embeddings

class SinsuodialPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # embedding vector dimension 512
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2 # sin, cosine을 사용하기 때문에 절반의 차원을 사용
        emb = math.log(10000) / (half_dim - 1) # 각 차원의 위치에 대해 서로 다른 주기를 가진 sin, cos 함수를 생성하는데 사용되는 sclaing factor
        emb = torch.exp(torch.arange(half_dim, device = device) * -emb) 
        emb = x[:, None] * emb[None, :] # input x의 각 위치에 대해 계산된 scaling factor를 곱함
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # 스케일링이 적용된 tensor에 sin, cos 함수를 적용하고, 이 두 tensor를 마지막 차원(dim=-1)을 따라 합침. 
        
        return emb


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out) # group normalization 
        self.act = nn.SiLU() # siwsh
        
    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)
        
        if exists(scale_shift): # helper functions 
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        # MLP는 time_emb_dim이 존재하면 initialize되고, 없으면 안됨
        self.mlp = nn.Sequential( # 시간 임베딩을 변환하는 multi layer perceptron
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity() # dim, dim_out이 같지 않으면 conv1d, 같을 경우 identity function
     
    def forward(self, x,time_emb= None):
         
        scale_shift = None # scale_shirt 변수 초기화, input x에 스케일링 및 이동 변환에 적용
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb) # 시간 임베딩 변환
            time_emb = rearrange(time_emb, 'b c -> b c 1') # 암베딩 변환, b = batch size, c = channel num, 시간 임베딩의 각 채널을 별도의 차원으로 변환해, 각 채널을 독립적으로 처리할 수 있도록 함.
            scale_shift = time_emb.chunk(2, dim = 1) # chunk: tensor를 여러 개(2개) 덩어리로 나눔. time_emb는 dim=1 을 기준으로 나뉨
        
        h = self.block1(x, scale_shift = scale_shift)
        
        h = self.block2(h)

        return h + self.res_conv(x)

# linear attention
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32): 
        super().__init__()
        self.scale = dim_head ** -0.5 # scaling factor, softmax에 적용하기 전에 query에 곱해짐. attention score의 분산을 조정하는데 사용
        self.head = heads 
        hidden_dim = dim_head * heads # 모든 attention head의 차원을 합친 것
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim) # helper module
        )
    
    def forward(self, x):
        b, c, n = x.shape # input tensor x shape으로 batch size, channel num, feature num 할당
        qkv = self.to_qkv(x).chunk(3, dim = 1) # input x 를 query, key, valuye로 변환
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv) # Q, K, V를 각각 self.heads 개수만큼의 head로 재배열
        
        q = q.softmax(dim = -2) # q에 softmax를 적용해 attention weight를 계산. q = 마지막에서 2번째 차원
        k = k.softmax(dim = -1) # k에 softmax를 적용해 attention weight를 계산. k = 마지막에서 1번째 차원
        
        q = q * self.scale # query에 scaling fact를 곱해 attention score의 분산을 조정
        
        context = torch.einsum('b h d n, b h e n -> b h d e') # key, value의 attention weight를 계산해 context vector를 얻음
        
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q) # output tensor의 형태를 변환, 각 head의 결과를 합쳐 원래의 dimension으로 돌려놓음
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads) # output tensor를 self.to_out(Conv1d와 RMSNorm으로 구성된 layer를 통해 변환하고, 이를 최종 결과로 변환)
        
        return self.to_out(out)
