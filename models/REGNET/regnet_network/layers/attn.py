import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f

class attnLink(nn.Module):
    def __init__(self, d_model, d_res, p2r=False, device=0):
        super(attnLink, self).__init__()
        self.input_size = d_model - d_res if p2r else d_res
        self.output_size = d_res if p2r else d_model - d_res
        self.device = device

        self.attn = nn.MultiheadAttention(embed_dim=self.input_size,num_heads=8)
        
        self.linear = nn.Linear(self.input_size, self.output_size)
        

        self.batchnorm = nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        # x (B, T, d1) -> out: (B, T, d2)
        x = x.permute(1,0,2) # (T, B, D1)
        z , _ = self.attn(x,x,x)
        out = self.linear(z)
        out = f.relu(out)

        return out.permute(1,0,2)