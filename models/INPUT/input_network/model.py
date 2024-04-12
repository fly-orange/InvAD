import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.fft as fft


class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        # simple block
        self.placeholder = nn.Conv1d(3,1,1)

    
    def forward(self, x): 
        '''return anomaly score and feature of point and window '''
        # x_in: (B, T, d)
        # x_f = fft.rfft(x, dim=1)
        scores = torch.mean(x, dim=-1) # (B, T)

        return scores
    

    def get_scores(self, x):
        
        d_scores = self.forward(x)
        w_scores = torch.max(d_scores, dim=-1)[0]
        
        return w_scores, d_scores

