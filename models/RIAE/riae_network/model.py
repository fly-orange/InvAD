import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.fft as fft


class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        # simple block
        
        self.encoder = nn.LSTM(input_size=configs.input_channel, hidden_size=64, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=64, hidden_size=configs.input_channel, num_layers=1, batch_first=True)
    
    def forward(self, x): 
        '''return anomaly score and feature of point and window '''

        # Encoder
        enc, _ = self.encoder(x)
        # Decoder
        output, _ = self.decoder(enc)
        # print(output.shape, x.shape)
        scores = (output - x)**2

        return torch.mean(scores, dim=-1)
    

    def get_scores(self, x):
        
        d_scores = self.forward(x)
        w_scores = torch.max(d_scores, dim=-1)[0]
        
        return w_scores, d_scores

