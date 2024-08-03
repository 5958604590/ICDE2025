import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from CATGNN.utils import utils


class GATv2(nn.Module):
    """
    An implementation of GATv2.
    For details about the algorithm see this paper:
    "How Attentive are Graph Attention Networks?"

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
        
        self.layers.append(dgl.nn.GATv2Conv(self.in_feats, 
                                            self.n_hidden, 
                                            self.heads[0], 
                                            residual=False, 
                                            activation=self.activation, 
                                            bias=False, 
                                            share_weights=True))
        
        for l in range(self.n_layers - 2):
            self.layers.append(dgl.nn.GATv2Conv(self.n_hidden * self.heads[l], 
                                                self.n_hidden, 
                                                self.heads[l+1], 
                                                residual=True,
                                                activation=self.activation, 
                                                bias=False, 
                                                share_weights=True))
        
        self.layers.append(dgl.nn.GATv2Conv(self.n_hidden * self.heads[-2], 
                                            self.n_classes, 
                                            self.heads[-1], 
                                            residual=True, 
                                            activation=None, 
                                            bias=False, 
                                            share_weights=True))

    # def forward(self, g, inputs):
    #     h = inputs
    #     for i, layer in enumerate(self.layers):
    #         h = layer(g, h)
    #         if i == self.n_layers-1:
    #             h = h.mean(1)
    #         else:
    #             h = h.flatten(1)
    #     return h
    
    
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[: block.num_dst_nodes()]
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        
        return h
    

    def inference(self, 
                  g, 
                  device, 
                  batch_size, 
                  num_workers, 
                  buffer_device=None):
        
        feat = g.ndata['feat']
        
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        
        dataloader = dgl.dataloading.DataLoader(g, 
                                                torch.arange(g.num_nodes()).to(g.device), 
                                                sampler, 
                                                device=device,
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                drop_last=False,
                                                num_workers=num_workers)

        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device, pin_memory=True)
            feat = feat.to(device)
            
            for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        
        return y