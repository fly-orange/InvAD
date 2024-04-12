import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f

class mlpLink(nn.Module):
    def __init__(self, d_model, d_res, p2r=False, device=0):
        super(mlpLink, self).__init__()
        self.input_size = d_model - d_res if p2r else d_res
        self.output_size = d_res if p2r else d_model - d_res
        self.link = nn.Sequential(nn.Linear(self.input_size, self.output_size), 
                                  nn.ReLU(), 
                                  nn.Linear(self.output_size, self.output_size)
                                  )

    def forward(self, x):
        # x (B, T, D)
        # print(x.shape)
        y = self.link(x)

        return y