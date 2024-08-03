import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from CATGNN.utils import utils


class SGC(nn.Module):
    """
    An implementation of Simple Graph Convolution (SGC).
    For details about the algorithm see this paper:
    " Simplifying Graph Convolutional Networks"

    Args:
        in_feats (int): Dimension of input features.
        n_classes (int): number of classes.
    """
    
    def __init__(self, 
                 in_feats: int = 32, 
                 n_classes: int = 32):
        super().__init__()
        
        self.in_feats = in_feats
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.SGConv(self.in_feats, 
                                         self.n_classes, 
                                         k=1, 
                                         cached=False, 
                                         bias=True))

    
    def forward(self, blocks, x):
        h = self.layers(blocks, x)
        return h

