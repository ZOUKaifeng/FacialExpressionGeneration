"""
Created on Mon Oct 15 13:43:10 2020

@Author: Kaifeng

@Contact: kaifeng.zou@unistra.fr

chebyshev conv and surface pooling for graph classification
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import probtorch
import logpdf

from math import sqrt

class ChebConv_batch(ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization=None, bias=True):
        super(ChebConv_batch, self).__init__(in_channels, out_channels, K, normalization, bias)

    def reset_parameters(self):
        normal(self.weight, 0, 0.1)
        normal(self.bias, 0, 0.1)


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, norm, edge_weight=None):
        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])
        x = x.transpose(0,1)
        Tx_0 = x
        if self.weight.size(0) > 1:

            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            Tx_1_transpose = Tx_1.transpose(0, 1)
            out = out + torch.matmul(Tx_1_transpose, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            Tx_2_transpose = Tx_2.transpose(0, 1)
            out = out + torch.matmul(Tx_2_transpose, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):

        return norm.view(-1, 1, 1) * x_j
        
class SurfacePool(MessagePassing):
    def __init__(self):
        super(SurfacePool, self).__init__(flow='target_to_source')

    def forward(self, x, pool_mat,  dtype=None):
        x = x.transpose(0,1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out.transpose(0,1)

    def message(self, x_j, norm):
        return norm.view(-1, 1, 1) * x_j

class cheb_AE(torch.nn.Module):

    def __init__(self, num_features, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes, model = 'MSE_VAE'):
        super(cheb_VAE, self).__init__()
        self.n_layers = config['n_layers']
        self.filters = list(config['num_conv_filters'])

        self.filters.insert(0, num_features)  # To get initial features per node
        self.K = config['polygon_order']

        
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_batch.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])

        # convolution layer
        self.cheb = torch.nn.ModuleList([ChebConv_batch(self.filters[i], self.filters[i+1], self.K[i])
                                         for i in range(len(self.filters)-2)])


        self.cheb_dec = torch.nn.ModuleList([ChebConv_batch(self.filters[-i-1], self.filters[-i-2], self.K[i])
                                             for i in range(len(self.filters)-1)])



        self.cheb_dec[-1].bias = None  # No bias for last convolution layer

        self.pool = SurfacePool()

        self.num_class = config['num_classes']



        self.num_hidden = config['z']



        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-1], self.num_hidden)


        self.dec_lin = torch.nn.Linear(self.num_hidden, self.downsample_matrices[-1].shape[0]*self.filters[-1] )

        self.reset_parameters()


    def forward(self, x):
        self.supervise = supervise
        # x = data
        
        batch_size = x.shape[0]

        x = x.reshape(batch_size, -1, self.filters[0])
        z = self.encoder(x)


        x = self.decoder(z)
    
       
        return x


    def encoder(self, x):
        for i in range(self.n_layers):
            x = F.relu(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            x = self.pool(x, self.downsample_matrices[i])
    

        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = self.enc_lin(x)
        return x

    def decoder(self, x):

        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):

            x = self.pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))

        recon_x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])

      
 
        return recon_x





