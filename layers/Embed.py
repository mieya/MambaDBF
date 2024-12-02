import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import sys



#######################################################################
class PatchEmbedding_DS(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super(PatchEmbedding_DS, self).__init__()
        # Patching
        self.patch_len = patch_len

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
#######################################################################

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# 根据频域动态划分patch
class DynamicPatchEmbedding(nn.Module):
    def __init__(self, d_model, dropout):
        super(DynamicPatchEmbedding, self).__init__()
        # Patching parameters will be set dynamically
        self.value_embedding = nn.Linear(1, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, top_k_freqs):
        # 动态调整 patch 大小，根据 top_k_freqs
        patch_len = int(max(1.0 / top_k_freqs.mean().item(), 1))
        stride = max(patch_len // 2, 1)  # 确保 stride 至少为 1

        x = F.pad(x, (0, patch_len), mode='replicate')
        x = x.unfold(dimension=-1, size=patch_len, step=stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x), x.shape[1]  # 返回 (patched_tensor, num_variables)
    
    
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
    
    
class TemporalEmbedding1(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding1, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # Adjust the shape to match [batch_size, n_vars, d_model]
        return (hour_x + weekday_x + day_x + month_x + minute_x)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:

            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)



class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x) # # [32, 8, 96] --> [32, 8, 512]
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) # [32, 8, 96] --> [32, 12, 512]
        # x: [Batch Variate d_model]
        return self.dropout(x)
    
class DataEmbedding_inverted_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted_pos, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x) # # [32, 8, 96] --> [32, 8, 512]
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) #+ self.position_embedding(x) # [32, 8, 96] --> [32, 12, 512]
        # x: [Batch Variate d_model]
        return self.dropout(x)

class DataEmbedding_inverted_UP(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted_UP, self).__init__()
        self.value_embedding = nn.Linear(c_in * 3, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x) # # [32, 8, 96] --> [32, 8, 512]
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) # [32, 8, 96] --> [32, 12, 512]
        # x: [Batch Variate d_model]
        return self.dropout(x)
    
    
class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # x = self.value_embedding(x) + self.position_embedding(x)
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

# class PatchEmbedding(nn.Module): 
#     def __init__(self, d_model, patch_len, stride, padding, dropout):
#         super(PatchEmbedding, self).__init__()
#         # Patching
#         self.patch_len = patch_len
#         self.stride = stride
#         self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

#         # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
#         self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

#         # Positional embedding
#         self.position_embedding = PositionalEmbedding(d_model)

#         # Residual dropout
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x): #[B, D, L]
#         # do patching
#         n_vars = x.shape[1]
#         x = self.padding_patch_layer(x) # 96填充到104
#         x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [32, 8, 12, 16]
#         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # [256, 12, 16] 12=patch_num 16=patch_len
        

#         # Input encoding
#         x = self.value_embedding(x) + self.position_embedding(x) [256, 12, 96]
#         return self.dropout(x), n_vars

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
    
# class PatchEmbedding(nn.Module):
#     def __init__(self, d_model, patch_len, stride, padding, dropout ,embed_type='fixed', freq='h'):
#         super(PatchEmbedding, self).__init__()
#         # Patching
#         self.patch_len = patch_len
#         self.stride = stride
#         self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

#         # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
#         self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

#         # Positional embedding
#         self.position_embedding = PositionalEmbedding(d_model)
        
#         self.temporal_embedding = TemporalEmbedding1(d_model=d_model, embed_type=embed_type,
#                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
#             d_model=d_model, embed_type=embed_type, freq=freq)

#         # Residual dropout
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, x_mark):
#         # do patching
#         n_vars = x.shape[1]
#         x = self.padding_patch_layer(x)
#         x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
#         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
#         if x_mark is None:
#                 x = self.value_embedding(x) + self.position_embedding(x)  # [batch_size * new_seq_len, n_vars, d_model]
#         else:
                
#             # 变换 x_mark 的形状以匹配 x 的形状
#             batch_size, seq_len, num_features = x_mark.shape
#             new_seq_len = (x.shape[0] // batch_size)  # 计算新的时间步长
                
#             # 调整 x_mark 形状
#             x_mark = x_mark.repeat(1, new_seq_len // seq_len + 1, 1)  # 扩展时间步长
#             x_mark = x_mark[:, :new_seq_len, :]  # 截取匹配的新时间步长
#             x_mark = torch.reshape(x_mark, (batch_size * new_seq_len, num_features))  # [batch_size * new_seq_len, num_features]
#             x_mark = x_mark.unsqueeze(1).repeat(1, n_vars, 1)  # [batch_size * new_seq_len, n_vars, num_features]
                
#             x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)  # [batch_size * new_seq_len, n_vars, d_model]

#         return self.dropout(x), n_vars

    
# class PatchEmbedding_inv(nn.Module):
#     def __init__(self, d_model, patch_len, stride, padding, dropout ,embed_type='fixed', freq='h'):
#         super(PatchEmbedding_inv, self).__init__()
#         # Patching
#         self.patch_len = patch_len
#         self.stride = stride
#         self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

#         # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
#         self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

#         # Positional embedding
#         self.position_embedding = PositionalEmbedding(d_model)

#         # Residual dropout
#         self.dropout = nn.Dropout(dropout)
        
#         self.temporal_embedding = TemporalEmbedding1(d_model=d_model, embed_type=embed_type,
#                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
#             d_model=d_model, embed_type=embed_type, freq=freq)
                                                    

#     def forward(self, x, x_mark):
#         # do patching
#         n_vars = x.shape[1]
#         x = self.padding_patch_layer(x)
#         x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [32, 7, 12, 16]
#         x = torch.reshape(x, (x.shape[0] * x.shape[2], x.shape[1], x.shape[3])) # [384, 7, 16]
        
#         if x_mark is None:
#             x = self.value_embedding(x)  + self.position_embedding(x)  # [192, 7, 512]
#         else:
#             # print(self.value_embedding(x).shape)
#             # print(self.temporal_embedding(x).shape)
#             # print(self.position_embedding(x) .shape)
#             # exit()
#             x = self.value_embedding(x)  +  self.temporal_embedding(x_mark) + self.position_embedding(x) 
            

#         # x = self.value_embedding(x)  + self.position_embedding(x)  # [192, 7, 512]
#         return self.dropout(x), n_vars
    


class PatchEmbedding_inv(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout ,embed_type='fixed', freq='h'):
        super(PatchEmbedding_inv, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
        self.temporal_embedding = TemporalEmbedding1(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
                                                    
    def forward(self, x, x_mark):
            # do patching
            n_vars = x.shape[1]
            x = self.padding_patch_layer(x)
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [batch_size, n_vars, new_seq_len, patch_len]
            x = torch.reshape(x, (x.shape[0] * x.shape[2], x.shape[1], x.shape[3])) # [batch_size * new_seq_len, n_vars, patch_len]

            if x_mark is None:
                x = self.value_embedding(x) + self.position_embedding(x)  # [batch_size * new_seq_len, n_vars, d_model]
            else:
                
                # 变换 x_mark 的形状以匹配 x 的形状
                batch_size, seq_len, num_features = x_mark.shape
                new_seq_len = (x.shape[0] // batch_size)  # 计算新的时间步长
                
                # 调整 x_mark 形状
                x_mark = x_mark.repeat(1, new_seq_len // seq_len + 1, 1)  # 扩展时间步长
                x_mark = x_mark[:, :new_seq_len, :]  # 截取匹配的新时间步长
                x_mark = torch.reshape(x_mark, (batch_size * new_seq_len, num_features))  # [batch_size * new_seq_len, num_features]
                x_mark = x_mark.unsqueeze(1).repeat(1, n_vars, 1)  # [batch_size * new_seq_len, n_vars, num_features]
                
                x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)  # [batch_size * new_seq_len, n_vars, d_model]

            return self.dropout(x), n_vars
    
    
class PatchEmbedding_(nn.Module): 
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding_, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): #[B, D, L] [32, 8, 96]
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x) # [32, 8, 108]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [32, 8, 9, 12] 
        x = torch.reshape(x, (x.shape[0] , x.shape[1] * x.shape[2],x.shape[3])) # [32, 72, 12] [B,N*patch_num,stride]
       # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) # [256, 9, 12]

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x) # [B,N*patch_num,d_model]
        return self.dropout(x), n_vars
    
class PatchEmbedding_woPosition(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding_woPosition, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)  
        return self.dropout(x), n_vars
    
