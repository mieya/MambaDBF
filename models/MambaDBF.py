import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted,PatchEmbedding
from layers.RevIN import RevIN
from mamba_ssm import Mamba_ffn
import torch.fft
from einops import rearrange
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)  
        x = self.linear(x)   
        x = self.dropout(x)
        return x


class TemPatch(nn.Module):
    def __init__(self, d_model, kernel_size=3, mlp_hidden_dim=128):
        super(TemPatch, self).__init__()
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        bsz, nvars, d_model, patch_num = x.shape
        x_residual = x  

        x = x.reshape(bsz * nvars, d_model, patch_num)
        x = self.conv1d(x)  
        x = x.reshape(bsz, nvars, d_model, patch_num)  
        x = self.layer_norm(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # Normalize after convolution

        return x + x_residual


class VarPatch(nn.Module):
    def __init__(self, patch_num, d_model, d_ff,d_state, d_conv, expand, dropout=0.3):
        super(VarPatch, self).__init__()
        self.mamba = Mamba_ffn(d_model=d_model, d_ff=d_ff, d_state=d_state, d_conv=d_conv, expand=expand)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        bsz, nvars, d_model, patch_num = x.shape
        x_residual = x
        
        x = x.reshape(bsz, patch_num * nvars, d_model)
        x = self.mamba(x)
        x = x.reshape(bsz, nvars, patch_num, d_model)
        x = self.layer_norm(x).permute(0, 1, 3, 2)
        return x  + x_residual
    
class Model(nn.Module):

    def __init__(self, configs , patch_len=16, stride=8):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.output_attention = configs.output_attention
        self.num_layers_e = configs.e_layers  
        self.num_layers_d = configs.d_layers
        padding = stride
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)

        self.tem_patch = nn.ModuleList([TemPatch(d_model=configs.d_model) for _ in range(configs.e_layers)])
        
        # Mamba
        self.var_patch = nn.ModuleList([VarPatch(patch_num=int((configs.seq_len - patch_len) / stride + 2),
                                                     d_model=configs.d_model, d_ff=configs.d_ff,d_state=configs.d_state,
                                                     d_conv=configs.dconv, expand=configs.e_fact) for _ in range(configs.e_layers)])

        self.layer_norm2 = nn.LayerNorm(configs.d_model)
        self.layer_norm3 = nn.LayerNorm(configs.d_model)
        
        self.proj = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self.revin_layer = RevIN(configs.enc_in)
        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,head_dropout=configs.dropout)

        
        # mamba
        self.mamba1 = nn.ModuleList([Mamba_ffn(d_model=configs.d_model,d_ff=16,d_state=configs.d_state,d_conv=configs.dconv,expand=configs.e_fact) for _ in range(configs.d_layers)])

        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, test_flag=0):
        B = x_enc.shape[0]
        N = x_enc.shape[2]
        
        x_enc = self.revin_layer(x_enc, 'norm') # (batch_size, seq_len, channels)

        # Variate-Embedding Branch
        enc_emb = self.enc_embedding(x_enc,x_mark=None)#.to('cuda:0')
        for layer in range(self.num_layers_d):
            enc_emb = self.mamba1[layer](enc_emb) 
        x_var = self.proj(enc_emb).permute(0,2,1) 
       
        # Patch-Embedding Branch
        enc_out, n_vars = self.patch_embedding(x_enc.permute(0,2,1)) 
        enc_out = rearrange(enc_out, "(b n) p d -> b n p d",n=n_vars)
        enc_out = enc_out.permute(0,1,3,2)

        enc_out_tem = enc_out
        for layer in range(self.num_layers_e):
            # Intra-patch 
            p_tem = self.tem_patch[layer](enc_out_tem)  
            # Inter-patch 
            p_var = self.var_patch[layer](p_tem)  
           
            if layer==0:
                p_var = self.layer_norm2(p_var.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            else:
                p_var = self.layer_norm3(p_var.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            enc_out_tem = p_var

        x_patch = self.head(enc_out_tem).permute(0,2,1) 

        enc_out = x_var  + x_patch 

        dec_out = self.revin_layer( enc_out , 'denorm')
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out