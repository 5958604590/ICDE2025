import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from CATGNN.utils import utils

"""An implementation of NGNN.
For details about the algorithm see this paper:
"Network In Graph Neural Network"

"""


class NGNN_GCNConv(torch.nn.Module):

    def __init__(self, 
                 input_channels, 
                 hidden_channels, 
                 output_channels):
        super(NGNN_GCNConv, self).__init__()
        self.conv = dgl.nn.GraphConv(input_channels, 
                                     hidden_channels)
        self.fc = nn.Linear(hidden_channels, 
                            output_channels)

    
    def forward(self, g, x, edge_weight=None):
        x = self.conv(g, x, edge_weight)
        x = F.relu(x)
        x = self.fc(x)
        
        return x



class NGNN_GCN(nn.Module):
    
    def __init__(self, 
                 input_channels, 
                 hidden_channels, 
                 output_channels):
        super(NGNN_GCN, self).__init__()
        
        self.conv1 = NGNN_GCNConv(input_channels, 
                                  hidden_channels, 
                                  hidden_channels)
        self.conv2 = dgl.nn.GraphConv(hidden_channels, 
                                      output_channels)

    
    def forward(self, g, input_channels):
        h = self.conv1(g, input_channels)
        h = F.relu(h)
        h = self.conv2(g, h)
        
        return h
