import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f

class rnnLink(nn.Module):
    def __init__(self, d_model, d_res, p2r=False, device=0):
        super(rnnLink, self).__init__()
        self.input_size = d_model - d_res if p2r else d_res
        self.output_size = d_res if p2r else d_model - d_res
        self.device = device
        kernel_size = 3

        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.output_size,
                           batch_first=True)
        
        self.linear = nn.Linear(self.output_size, self.output_size)
        

        self.batchnorm = nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        # x (B, T, d1) -> out: (B, T, d2)
        # print(x.shape)

        z,_ = self.rnn(x) # (B, T, D1)
        out = self.linear(z)
        out = self.batchnorm(out.permute(0,2,1)).permute(0,2,1)
        out = f.relu(out)

        return out