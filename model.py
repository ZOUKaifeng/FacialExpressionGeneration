
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import matplotlib.pyplot as plt

from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp
import  open3d as o3d
from scipy.linalg import orthogonal_procrustes

import math


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
  
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        
        
        mask = mask.unsqueeze(1).repeat(1,4, 1, scores.shape[-1])

        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, dim_model)
        self.fc_K = nn.Linear(dim_model, dim_model)
        self.fc_V = nn.Linear(dim_model, dim_model)
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(dim_model)

    def forward(self, q, k, v, mask = None):
        batch_size = q.size(0)
        Q = self.fc_Q(q)
        K = self.fc_K(k)
        V = self.fc_V(v)
        Q = Q.view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)

        context, _ = attention(Q, K, V, mask = mask)
       
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)


        return out


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class TransformerEncoder(nn.Module):
    def __init__(self, encode_unit, latent_size, num_head, dropout = 0.3):
        super(TransformerEncoder, self).__init__()
        

        

        self.layer_norm = nn.LayerNorm(encode_unit, eps=1e-6)
        self.attention = Multi_Head_Attention(encode_unit ,num_head)
        self.feed_forward = PositionwiseFeedForward(encode_unit, ff_size=encode_unit*4,
                                                    dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.out_layer_norm = nn.LayerNorm(encode_unit, eps=1e-6)

    def forward(self, x, mask = None):
       
        x_norm = self.layer_norm(x)
        att = self.attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(att)
        x_norm = self.out_layer_norm(x)
        x = self.feed_forward(x_norm)
        x = x + self.dropout(x)

        return x



class TransformerDecoder(nn.Module):

    def __init__(self, num_frame, decode_unit, num_head, dropout = 0.3):
        super(TransformerDecoder, self).__init__()
        
        
        self.inputs_layer_norm = nn.LayerNorm(decode_unit, eps=1e-6)

        self.target_att = Multi_Head_Attention(decode_unit ,num_head)
        self.source_att = Multi_Head_Attention(decode_unit ,num_head)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(decode_unit, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(decode_unit, ff_size=decode_unit*4,
                                                dropout=dropout)
    def forward(self,inputs, z, mask = None):
        
        
        inputs = self.inputs_layer_norm(inputs)
        targ_att = self.target_att(inputs, inputs, inputs, mask)
        x = inputs + self.dropout(targ_att)
        x = self.layer_norm(x)
        sourc_att = self.source_att(x, z, z)

        out = self.feed_forward(self.dropout(sourc_att) + x)
        return out








class DisenFace(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_pts, num_frame, label_size, hidden_unit,  latent_size, num_head, num_layers=3, dropout = 0.1):
        super(DisenFace, self).__init__()
        self.embedding_layer = nn.Linear(num_pts, hidden_unit)
        self.num_layers = int(num_layers)
        self.encoder = nn.ModuleList([TransformerEncoder(hidden_unit, latent_size, num_head, dropout ) for i in range(self.num_layers)])
        self.decoder = nn.ModuleList([TransformerDecoder( num_frame, hidden_unit, num_head, dropout) for i in range(self.num_layers)])
        self.latent_size = latent_size
        self.num_frame = num_frame
        self.final_layer = nn.Linear(hidden_unit, num_pts)
        self.z_mean = nn.Linear(hidden_unit, hidden_unit)
        self.log_sigma = nn.Linear(hidden_unit, hidden_unit)
        #self.z_linear = nn.Linear(latent_size+label_size, hidden_unit)
        self.actionBiases = nn.Parameter(torch.randn(label_size, hidden_unit))
        self.position_encoding = PositionalEncoding(hidden_unit, dropout)
        self.num_frame = num_frame
        self.hidden_unit = hidden_unit
    def forward(self, x, one_hot_y, mask, train = True):
        results = {}
        bs = x.shape[0]
        x = self.embedding_layer(x)
        timequeries = x.clone()
        x = self.position_encoding(x)
       
        for i in range(self.num_layers):
            x = self.encoder[i](x)

        h = x.mean(1)
        z_mean = self.z_mean(h)
        log_sigma = self.log_sigma(h)
        
        results['mean'] = z_mean
        results['log_sigma'] = log_sigma

        if train:
            z = self.reparameterize(results['mean'], results['log_sigma'])
        else:
            z = results["mean"]

        index = torch.argmax(one_hot_y, dim = 1)
        z = torch.stack((z, self.actionBiases[index]), axis=1)
     #   z = torch.cat([z, one_hot_y], -1)

       # z = self.z_linear(z)

      #  timequeries = torch.zeros(bs,  self.num_frame, self.hidden_unit, device=z.device, requires_grad = True)
        x = self.position_encoding(timequeries)
        for i in range(self.num_layers):
            x = self.decoder[i](x, z)
      #  x = self.decoder(x, z, mask)

        out = self.final_layer(x)
        results["x"] = out

        return results 

    def reparameterize(self, mu, logvar):

        batch_size = mu.shape[0]
        dim = logvar.shape[-1]

        std = torch.exp(logvar * 0.5)
  
        z = torch.normal(mean = 0, std = 1, size=(batch_size, dim)).to(mu.device)
        z = z*std + mu

    

        return z

    def sample(self, z, one_hot_y, mask):
        bs = z.shape[0]
        index = torch.argmax(one_hot_y, dim = 1)
        z = torch.stack((z, self.actionBiases[index]), axis=1)

        timequeries = torch.zeros(bs,  self.num_frame, self.hidden_unit, device=z.device, requires_grad = True)
        x = self.position_encoding(timequeries)
        for i in range(self.num_layers):
            x = self.decoder[i](x, z, mask)

        out = self.final_layer(x)

        return out
