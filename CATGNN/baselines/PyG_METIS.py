import sys
import os
import torch
from torch_geometric.data import Data
from torch_geometric.distributed import Partitioner
from timeit import default_timer as timer
import pandas as pd

dataset = 'cora'
print(dataset)

path = os.path.abspath(os.getcwd())

edge_list = pd.read_csv(path + '/datasets/' + dataset + '/edge_list.csv', header=None)

edge_index = torch.tensor(edge_list.T.values, dtype=torch.long)

data = Data(x=None, edge_index=edge_index)

number_partition = 4
print('number_partition: ', number_partition)

partitioner = Partitioner(data, number_partition, path + '/output/' + 'partitions-' + str(number_partition) + '/' + dataset  + '/', recursive=False)
partitioner.generate_partition()
print('==============================')
