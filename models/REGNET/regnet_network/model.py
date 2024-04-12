import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.fft as fft
from einops import rearrange
from .embed import DataEmbedding, DataEmbedding_wo_pos, FreqEmbedding
from .inn import INN
from .softdtw_cuda import SoftDTW
from .attn import AnomalyAttention, AttentionLayer

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = f.relu if activation == "relu" else f.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        self.output_attention = configs.output_attention
        dropout = 0.0
        # Encoding
        self.embedding = DataEmbedding(configs.input_channels, configs.d_model, dropout)

        # Encoder


    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]


class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        # simple block
        self.temp_embed = DataEmbedding_wo_pos(configs.input_channel, configs.d_model) # (T, D)
        # self.freq_embed = FreqEmbedding(configs.input_channel, configs.d_model) # (T, D)
        self.d_res = configs.d_res
        dropout = 0.0
        # INN encoder
        # self.inv_net = INN(configs.d_model, configs.d_res, configs.link, configs.num_inv, configs.clamp, device)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(configs.window_size, False, attention_dropout=dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # score net
        self.pooling_type = configs.pooling_type
        self.fea_net = nn.Linear(configs.d_res, configs.d_model)
        self.score_net = nn.Linear(configs.d_model, 1)
        self.rec_criterion = nn.CosineSimilarity(dim=-1)
        self.recoef = configs.recoef
        
        self.projection = nn.Linear(configs.d_model, configs.d_model, bias=True)

        # configs
        self.granularity = configs.granularity
        self.local_threshold = configs.local_threshold
        self.global_threshold = configs.global_threshold
        self.dtw = SoftDTW(use_cuda=True, gamma=configs.gamma, normalize=False)


    
    def forward(self, x): 
        '''return anomaly score and feature of point and window '''
        # x_in: (B, T, d)
        # x_f = fft.rfft(x, dim=1)
        emb_temp = self.temp_embed(x, None) # (B, T, D)

        enc,_,_,_ = self.encoder(emb_temp) # (B, T, D)
        
        enc_res  = enc[:,:, :-self.d_res]
        enc_pri = enc[:,:, -self.d_res:]
        
        rec_temp = self.projection(enc)

        # fea_t = self.fea_net(torch.cat(enc_temp, enc_freq),dim=-1)
        fea_t = self.fea_net(enc_pri)
        # print(fea_t)
        return fea_t, enc_res, rec_temp, emb_temp
    
    def get_cls_scores(self, fea_t):

        if self.pooling_type == 'avg':
            fea_w = torch.mean(fea_t, dim=1)
        elif self.pooling_type == 'max':
            fea_w = torch.max(fea_t, dim=1)[0] # (B, D)

        # drec_loss = self.rec_criterion(rec_temp, emb_temp) # (-1, 1)
        # drec_loss = (drec_loss + 1) / 2 # (0, 1)
        # wrec_loss = torch.mean(drec_loss, dim=-1)

        wfeat = fea_w
        # ret['wscore'] = (1 - self.recoef) * torch.sigmoid(self.score_net(fea_w).squeeze(dim=1)) + self.recoef * wrec_loss 
        wscore = torch.sigmoid(self.score_net(fea_w).squeeze(dim=1))
        wpred = (wscore >= self.global_threshold).type(torch.cuda.FloatTensor)

        dfeat = fea_t
        # ret['dscore'] = (1 - self.recoef) * torch.sigmoid(self.score_net(fea_t)).squeeze(dim=2) + self.recoef * drec_loss  # (B, T)
        dscore = torch.sigmoid(self.score_net(fea_t).squeeze(dim=2))
        dpred = (dscore >= self.local_threshold).type(torch.cuda.FloatTensor)
        
        return (wfeat, wscore, wpred, dfeat, dscore, dpred)
    
    def get_rec_scores(self, enc_res, rec_temp, emb_temp):
        self.ILoss = nn.MSELoss(reduction='none')
        rec_score = torch.mean(self.ILoss(rec_temp, emb_temp),dim=-1) # (B, T)
        # rec_score += torch.mean(self.ILoss(enc_res, torch.ones_like(enc_res)*self.res_const), dim=-1) 
        # rec_score = f.sigmoid(rec_score)
        
        return rec_score
    

    def get_scores(self, x, cls=True, rec=False):
        fea_t, enc_res, rec_temp, emb_temp = self.forward(x)
        if not cls and not rec:
            raise ValueError('At least one type of anomaly score needed')
        elif cls and rec:
            return self.get_cls_scores(fea_t), self.get_rec_scores(enc_res, rec_temp, emb_temp)
        elif cls:
            return self.get_cls_scores(fea_t)
        else:
            return self.get_rec_scores(enc_res, rec_temp, emb_temp)


    def get_seqlabel(self, actmap, wlabel):   # 若L中存在异常（超过某一阈值），则该段为异常
        actmap *= wlabel.unsqueeze(dim=1).repeat(1, actmap.shape[1]) # filter those normal sample (B, T)
        seqlabel = (actmap >= self.local_threshold).type(torch.cuda.FloatTensor) # (B, T)
        seqlabel = f.pad(seqlabel, (self.granularity - seqlabel.shape[1] % self.granularity, 0), 'constant', 0)
        seqlabel = torch.reshape(seqlabel, (seqlabel.shape[0], -1, int(seqlabel.shape[1] / self.granularity))) # (B, L, T/L)
        seqlabel = torch.max(seqlabel, dim=2)[0] # (B, L)

        seqlabel = torch.cat([torch.zeros(seqlabel.shape[0], 1).cuda(), seqlabel, torch.zeros(seqlabel.shape[0], 1).cuda()], dim=1) # (B, L+2)

        return seqlabel
    
    def get_alignment(self, label, score):
            # label : batch x pseudo-label length (= B x L)
        # score : batch x time-sereis length (= B x T)
        assert label.shape[0] == score.shape[0]
        assert len(label.shape) == 2  
        assert len(score.shape) == 2

        A = self.dtw.align(label.unsqueeze(dim=1), score.unsqueeze(dim=1))
        indices = torch.max(A, dim=1)[1]
        return torch.gather(label, 1, indices)

    def get_dpred(self, out, wlabel):
        '''out: (B, T, d)'''
        # indexes = self.indexes.unsqueeze(0).unsqueeze(-1).repeat(out.size(0), 1, 1, out.size(2)).to(out.device) # (B, T, s, D)
        # indexes = indexes.view(out.size(0), -1, out.size(2))  # (B, T*s, D)
        # all_enc = torch.gather(out, 1, indexes) # select feature for score (B, T*s)
        # all_enc = all_enc.view(out.size(0), self.effe_size[0], -1, out.size(2)) # (B, T, s, D)
        # enc = torch.mean(all_enc, dim=2) # (B, T, D)
                
        d_h = self.score_net(out).squeeze(dim=2) # (B, T)
        dscore = torch.sigmoid(d_h)

        # rec_loss = self.rec_criterion(rect, embt)
        # dscore = (1 - self.recoef) * dscore + self.recoef * rec_loss

        # self.index: (T, s)
        # indexes = self.indexes.unsqueeze(0).repeat(h.size(0), 1, 1).to(h.device) # (B, T, s)
        # indexes = indexes.view(h.size(0), -1)  # (B, T*s, D)
        # all_enc = torch.gather(h, 1, indexes) # select feature for score (B, T*s)
        # all_enc = all_enc.view(h.size(0), self.effe_size[0], -1) # (B, T, s)
        
        # d_h = torch.sum(all_enc, dim=-1) # (B, T)
        # # dscore = torch.sum(all_enc, dim=-1)
        # dscore = torch.sigmoid(d_h)
        # d_h = h[:,:self.seq_size]
        
        with torch.no_grad():
            # Activation map
            actmap = d_h
            actmin = torch.min(actmap, dim=1)[0]
            actmap = actmap - actmin.unsqueeze(dim=1)
            actmax = torch.max(actmap, dim=1)[0]
            actmap = actmap / actmax.unsqueeze(dim=1)
            # Sequential labels
            seqlabel = self.get_seqlabel(actmap, wlabel)

        return self.get_alignment(seqlabel, dscore)
    




    



