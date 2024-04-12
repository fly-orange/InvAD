import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation):
        super(ResidualBlock, self).__init__()

        self.dilation = dilation

        self.diconv = nn.Conv1d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                dilation=dilation)

        self.conv1by1_skip = nn.Conv1d(in_channels=output_size,
                                       out_channels=output_size,
                                       kernel_size=1,
                                       dilation=1)

        self.conv1by1_out = nn.Conv1d(in_channels=output_size,
                                      out_channels=output_size,
                                      kernel_size=1,
                                      dilation=1)

    def forward(self, x):
        x = f.pad(x, (self.dilation, 0), "constant", 0)
        z = self.diconv(x)
        z = torch.tanh(z) * torch.sigmoid(z)
        s = self.conv1by1_skip(z)
        z = self.conv1by1_out(z) + x[:,:,-z.shape[2]:]
        return z, s

class DiCNNLink(nn.Module):
    def __init__(self, d_model, d_res, p2r=False, device=0):
        super(DiCNNLink, self).__init__()
        self.input_size = d_model - d_res if p2r else d_res
        self.output_size = d_res if p2r else d_model - d_res
        self.device = device
        hidden_size = 32
        kernel_size = 2
        n_layers = 2

        self.rf_size = kernel_size ** n_layers
        self.causal_conv = nn.Conv1d(in_channels=self.input_size,
                                     out_channels=hidden_size,
                                     kernel_size=kernel_size,
                                     stride=1, dilation=1)

        # dilated conv. layer
        self.diconv_layers = nn.ModuleList()
        for i in range(n_layers):
            diconv = ResidualBlock(input_size=hidden_size,
                                   output_size=hidden_size,
                                   kernel_size=kernel_size,
                                   dilation=kernel_size**i)
            self.diconv_layers.append(diconv)

        # 1x1 conv. layer (for skip-connection)
        self.conv1by1_skip1 = nn.Conv1d(in_channels=hidden_size,
                                        out_channels=hidden_size,
                                        kernel_size=1, dilation=1)

        self.conv1by1_skip2 = nn.Conv1d(in_channels=hidden_size,
                                        out_channels=self.output_size,
                                        kernel_size=1, dilation=1)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x (B, T, d1) -> out: (B, T, d2)
        # print(x.shape)
        x = x.transpose(1, 2) # (B, D, T)

        padding_size = self.rf_size - x.shape[2]
        if padding_size > 0:
            x = f.pad(x, (padding_size, 0), "constant", 0)
        
        x = f.pad(x, (1, 0), "constant", 0)
        z = self.causal_conv(x) # (B, D, T)

        out = torch.zeros(z.shape).to(self.device)
        for diconv in self.diconv_layers:
            z, s = diconv(z)
            out += s

        out = f.relu(out)
        out = self.conv1by1_skip1(out)
        out = f.relu(out)
        out = self.conv1by1_skip2(out).transpose(1, 2) # (B, T, D)
        return out
    
class tcnLink(nn.Module):
    def __init__(self, d_model, d_res, p2r=False, device=0):
        super(tcnLink, self).__init__()
        self.input_size = d_model - d_res if p2r else d_res
        self.output_size = d_res if p2r else d_model - d_res
        self.device = device
        kernel_size = 3

        self.causal_conv = nn.Conv1d(in_channels=self.input_size,
                                     out_channels=self.output_size,
                                     kernel_size=kernel_size,
                                     stride=1, dilation=1)
        self.conv1by1_skip = nn.Conv1d(in_channels=self.output_size,
                                       out_channels=self.output_size,
                                       kernel_size=1,
                                       dilation=1)

        self.batchnorm = nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        # x (B, T, d1) -> out: (B, T, d2)
        # print(x.shape)
        x = x.transpose(1, 2) # (B, D, T)
        
        x = f.pad(x, (1, 1), "constant", 0)

        z = self.causal_conv(x) # (B, D, T)
        z = torch.tanh(z) * torch.sigmoid(z)
        out = self.conv1by1_skip(z)
        
        out = self.batchnorm(out)
        out = f.relu(out)

        return out.transpose(1, 2)
    
