import torch
import torch.nn as nn
from .layers import mlpLink, attnLink, tcnLink, rnnLink, DiCNNLink

subnet_family={
    'mlp':  mlpLink,
    'attn': attnLink,
    'tcn':  tcnLink,
    'rnn':  rnnLink,
    'dicnn': DiCNNLink
}

class INN(nn.Module):
    
    def __init__(self, d_model, d_res, link, num_inv, clamp, device):
        super(INN, self).__init__()
        self.num_inv = num_inv
        self.inv_net = nn.ModuleList([INN_block(d_model, d_res, link, clamp, device) for _ in range(num_inv)])


    def forward(self, x, y, rev=False):

        if not rev:
            for i in range(self.num_inv):
                x,y = self.inv_net[i](x,y, rev=False)

        else:
            for i in range(self.num_inv):
                x,y = self.inv_net[i](x,y, rev=True)

        return x, y
    
class INN_block(nn.Module):
    def __init__(self, d_model, d_res, link='attn', clamp=0.2, device=0):
        super().__init__()
        self.clamp = clamp

        # ρ

        self.r = subnet_family[link](d_model, d_res, True, device)
        # η
        self.y = subnet_family[link](d_model, d_res, True, device)
        # φ
        self.f = subnet_family[link](d_model, d_res, False, device)
    
    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    
    def forward(self, x1, x2, rev=False):
        '''
         x1: (B, T, d - d_res), x2: (B, T, d_res)
         x2 --------------------------------+--------- y2
                  |           |             |
                  |           |             |
                f(x2)        e(y1)        y(y1)
                  |           |             |
                  |           |             |
         x1 ------+----------------------------------- y1 
        '''
        #print(x1.shape)
        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1
            # y1 = self.f(x2) + x1
            # y2 = self.y(y1) + x2

        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)
            # y2 = x2 - self.y(x1)
            # y1 = x1 - self.f(y2)

        return y1, y2  # (B, T, 2D)