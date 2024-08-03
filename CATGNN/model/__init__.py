from .graphsage import GraphSAGE
from .gat import GAT
from .gcn import GCN
from .graphsaint import GraphSAINT
from .gatv2 import GATv2
from .clustergcn import ClusterGCN
from .sgc import SGC
from .ngnn import NGNN_GCN
from .graphsage_full import GraphSAGE_Full
from .gat_full import GAT_Full
from .gcn_full import GCN_Full
from .custom_gnn import CustomGNN


__all__ = [
    "GraphSAGE",
    "GAT",
    "GCN",
    "GraphSAINT",
    "ClusterGCN",
    "GATv2",
    "SGC",
    "NGNN_GCN",
    "CustomGNN",
    "GraphSAGE_Full",
    "GAT_Full",
    "GCN_Full"
]

classes = __all__