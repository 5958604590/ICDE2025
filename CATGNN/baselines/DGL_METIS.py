import sys
import os
import torch
import dgl
import pandas as pd

dataset = 'cora'
print(dataset)

path = os.path.abspath(os.getcwd())

edge_list = pd.read_csv(path + '/datasets/' + dataset + '/edge_list.csv', header=None, names = ['src','dst'], index_col=False)
src = edge_list['src'].to_list()
dst = edge_list['dst'].to_list()

graph = dgl.DGLGraph()
graph.add_edges(src, dst)

number_partition = 4
print('number_partition: ', number_partition)

partitions = dgl.metis_partition_assignment(graph, number_partition, balance_edges=True, mode='k-way', objtype='cut')

partition_graph, node_map, _ = dgl.partition_graph_with_halo(graph, partitions, extra_cached_hops=1, reshuffle=True)

print('==============================')
