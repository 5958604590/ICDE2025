import numpy as np
import pandas as pd
import random
import json
import pickle
import os
import csv
import gzip
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class Partitioner(object):
    """
    Partitioner base class with constructor and private methods.
    """
    
    def __init__(self):
        pass


    def _set_seed(self):
        """Creating the initial random seed."""
        
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    
    def get_degree(self):
        """Compute node degree."""

        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
        reader = csv.reader(self.file, delimiter=',')
        
        for number_edges, line in enumerate(reader):
            i, j = int(line[0]), int(line[1])
            self.node_degree[int(j)] += 1

        self.number_edges = number_edges + 1
        self.number_nodes = max(self.node_degree) + 1
        print('Number of nodes: ', self.number_nodes)
        print('Number of edges: ', self.number_edges)

    
    def partition(self):
        """Partition a graph."""
        pass


    def partition_features(self):
        """Partition the features of a graph."""
        
        node_feats = np.load(self.path + self.dataset +'/feats.npy', mmap_mode='r')
        labels = json.load(open(self.path + self.dataset + '/class_map.json'))
        node_labels = np.array(list(labels.values()), dtype=np.int64)

        if self.multilabel:
            n_classes = len(node_labels[0])
        else:
            n_classes = len(np.unique(node_labels))
        
        with open(self.output_path + 'num_classes.txt', 'w') as f:
            f.write(str(n_classes))

        n_feats = node_feats.shape[1]
        
        with open(self.output_path + 'num_feats.txt', 'w') as f:
            f.write(str(n_feats))
        
        for i in range(self.number_partition):
            edge_list = np.loadtxt(self.output_path + 'partition_' + str(i) + '.txt', dtype=int)
            node_set = np.unique(edge_list)
            feat = node_feats[node_set]
            np.save(self.output_path + 'partition_' + str(i) + '-feats.npy', feat)

            label = node_labels[node_set]
            np.save(self.output_path + 'partition_' + str(i) + '-labels.npy', label)
        
        
    def partition_statistics(self):
        """Get the partitioning statistics."""
        
        # n_nodes_partitions = Counter(self.v2p.values())
        # print('number of nodes of each partition: ', n_nodes_partitions)

        role = json.load(open(self.path + self.dataset + '/role.json'))
        self.train_ids = role['tr']
        
        # print('number of training node: ', len(self.train_ids))
        with open(self.output_path + 'partition' + '.json', 'rb') as fp:
            node_partition_dict = pickle.load(fp)
        
        n_nodes_list = []
        for k in range(self.number_partition):
            
            # train_node_p = [j for j in self.train_ids if node_partition_dict[j]==k]
            # val_node_p = [j for j in val_ids if node_partition_dict[j]==k]
            # test_node_p = [j for j in test_ids if node_partition_dict[j]==k]
            
            with open(self.output_path + 'partition_' + str(k) + '.txt', 'r') as file:
                n_edges = 0
                node_set = set()
                for line in file:
                    n_edges += 1
                    i, j = line.strip().split(' ')
                    node_set.add(int(i))
                    node_set.add(int(j))

            n_nodes = len(node_set)
            # print('Partition ', k)
            # print('number of nodes: ', n_nodes)
            # print('number of edges: ', n_edges)
            # print('number of training nodes: ', len(set(self.train_ids) & node_set))
            # print('number of training nodes: ', len(set(train_node_p) & node_set))
            # print('='*30)
            n_nodes_list.append(n_nodes)
            
        rf = sum(n_nodes_list)/self.number_nodes
        print('Replication Factor: ', rf)

    
    def run(self):
        """Run the partitioning algorithm."""
        self.partition()
        
        if self.partition_features_file:
            self.partition_features()
            
        if self.print_partition_statistics:
            self.partition_statistics()