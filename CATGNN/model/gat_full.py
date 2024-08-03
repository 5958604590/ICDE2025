import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from CATGNN.utils import utils


class GAT_Full(nn.Module):
    """
    An implementation of full batch GAT.
    For details about the algorithm see this paper:
    "Graph Attention Networks"

    Args:
        in_feats (int): Dimension of input features.
        n_hidden (int): Dimension of hidden layers.
        n_layer (int): Number of hidden layers.
        n_classes (int): Number of classes.
        n_heads (int): Number of heads in Multi-Head Attention.
        n_out_heads (int): Number of output heads.
        activation (str): 'relu', 'sigmoid', 'tanh', 'leaky_relu'.
        feat_dropout (float, optional): Dropout rate on feature. Defaults: 0.
        attn_dropout (float, optional): Dropout rate on attention weight. Defaults: 0.
        negative_slope (float, optional): LeakyReLU angle of negative slope. Defaults: 0.2.
        residual (bool, optional): If True, use residual connection. Defaults: False.
    """
    
    def __init__(self, 
                 in_feats: int = 32, 
                 n_hidden: int = 32, 
                 n_layers: int = 2, 
                 n_classes: int = 32, 
                 heads: list = [2,2,2],
                 feat_dropout: float = 0.0, 
                 attn_dropout: float = 0.0, 
                 negative_slope: float = 0.2, 
                 residual: bool = False, 
                 activation: str = 'relu'):
        super().__init__()
        
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_classes = n_classes
        # self.n_heads = n_heads
        # self.n_out_heads = n_out_heads
        # self.heads = ([self.n_heads] * self.n_layers) + [self.n_out_heads]
        self.heads = heads
        self.feat_dropout = feat_dropout
        self.attn_dropout = attn_dropout
        self.negative_slope = negative_slope
        self.residual = residual
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == None:
            self.activation = activation
        else:
            raise NotImplementedError('No Support for \'{}\' Yet. Please Try Different Activation Functions.'.format(activation))
        
        
        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.GATConv(self.in_feats, 
                                          self.n_hidden, 
                                          self.heads[0], 
                                          activation=self.activation))
        
        for l in range(self.n_layers - 2):
            self.layers.append(dgl.nn.GATConv(self.n_hidden * self.heads[l], 
                                              self.n_hidden, 
                                              self.heads[l+1], 
                                              residual=True, 
                                              activation=self.activation))
        
        self.layers.append(dgl.nn.GATConv(self.n_hidden * self.heads[-2], 
                                          self.n_classes, 
                                          self.heads[-1], 
                                          residual=True, 
                                          activation=None))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i == self.n_layers-1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h
    