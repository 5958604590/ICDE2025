import numpy as np
import pandas as pd
import random
import json
import pickle
import os
import csv
import gzip
from collections import defaultdict,  Counter
import warnings
warnings.filterwarnings('ignore')
# from memory_profiler import profile
from CATGNN.partition.partitioner import Partitioner
import pprint

class Clustering(Partitioner):
    """
    An implementation of Clustering method.
    For details about the algorithm see this paper:
    "A streaming algorithm for graph clustering"

    Args:
        dataset (str): Dataset name.
        path (str): Dataset root path.
        output_path (str): Output path.
        number_partition (int): Number of partitions. Default is auto.
        stream_iters (int): Number of streaming iterations for clustering.
        K (int): Number of hops of neighbor maintained after partitioning. Default is 1.
        seed (int): Random seed. Default is 42.
        partition_features_file (bool): Partition the features file if True,
        print_partition_statistics (bool): Print out the statistics of the partitioned graph if True.
    """

    def __init__(self, 
                 dataset: str = None, 
                 multilabel: bool=False, 
                 path: str = None, 
                 output_path: str = None, 
                 number_partition: int = 4, 
                 K: int = 1, 
                 stream_iters: int =1, 
                 seed: int = 42,
                 partition_features_file: bool = True,
                 print_partition_statistics: bool = True):
        super().__init__()
        
        self.dataset = dataset
        self.multilabel = multilabel
        self.path = path
        self.output_path = output_path
        self.number_partition = number_partition
        self.K = K
        self.seed = seed
        self._set_seed()
        self.node_degree = defaultdict(int)
        self.stream_iters = stream_iters
        
        self.partition_features_file = partition_features_file
        self.print_partition_statistics = print_partition_statistics


    def do_clustering(self, i, j, c, v):
        if c[i] == 0:
            c[i] = self.next_c_id
            v[c[i]] += self.node_degree[i]
            self.next_c_id += 1
        if c[j] == 0:
            c[j] = self.next_c_id
            v[c[j]] += self.node_degree[j]
            self.next_c_id += 1
        if v[c[i]] <= self.v_max and v[c[j]] <= self.v_max:
            if v[c[i]] <= v[c[j]]:
                v[c[j]] += self.node_degree[i]
                v[c[i]] -= self.node_degree[i]
                c[i] = c[j]
            else:
                v[c[i]] += self.node_degree[j]
                v[c[j]] -= self.node_degree[j]
                c[j] = c[i]
                

    def restream_clustering(self):
        self.v2c = defaultdict(int)
        self.vol = defaultdict(int)
        self.n_training = defaultdict(int)
        self.next_c_id = 1
        self.v_max = 0.1*self.number_edges/self.number_partition
        role = json.load(open(self.path + self.dataset + '/role.json'))
        self.train_ids = role['tr']
        
        for _ in range(self.stream_iters):
            self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
            reader = csv.reader(self.file, delimiter=',')
            for _, line in enumerate(reader):
                i, j = int(line[0]), int(line[1])
                self.do_clustering(i, j, self.v2c, self.vol)
            
        # print('v2c', self.v2c)
        # print('vol', self.vol)
        com_dict = defaultdict(set)
        for i in self.v2c:
            com_dict[self.v2c[i]].add(i)
        self.communities = list()
        for com in com_dict.values():
            if len(com) >= 1:
                self.communities.append(com)

        # print("com_dict: ", com_dict)
        # print("# of clusters: ", len(self.communities))
        # print('communities', self.communities)

        #create self.n_training with size of len(self.communities)
        self.n_training = []
        for i in range(len(self.communities)):
            self.n_training.append(len(self.communities[i] & set(self.train_ids)))
        # print("n_training", self.n_training)
        
        # size_clusters = Counter(self.v2c.values())
        # print('size_clusters: ', size_clusters)
        return self.communities

    def cluster2partition(self):
        sort_c = sorted(self.communities, key=len, reverse=True)
        # sort_c = [x for x, _ in sorted(zip(self.communities, self.n_training), reverse=True)]

        self.list_p = [list() for i in range(self.number_partition)]
        list_p_size = np.array([0 for i in range(self.number_partition)])

        for i in range(len(sort_c)):
            idx = np.argmin(list_p_size)
            self.list_p[idx] += sort_c[i]
            list_p_size[idx] += len(sort_c[i])

        # print("# of nodes for each partition: ", list_p_size)
        # print('list_p, ', self.list_p)
        return self.list_p

    def partition(self):
        """Partition a graph."""

        for i in range(self.number_partition):
            with open(os.path.join(self.output_path, 'partition_' + str(i) + '.txt'), 'w') as f:
                pass
        
        self.get_degree()
        self.restream_clustering()
        self.cluster2partition()
        self.v2p = defaultdict(int)
        for i in range(len(self.list_p)):
            for j in self.list_p[i]:
                self.v2p[j] = i
        
        with open(self.output_path + 'partition' + '.json', 'wb') as json_file:
            pickle.dump(self.v2p, json_file)

        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
          
        file_objects = []
        for i in range(self.number_partition):
            filename = os.path.join(self.output_path, 'partition_' + str(i) + '.txt')
            files = open(filename, 'a', newline='')
            # Append the file object to the list
            file_objects.append(files)
        
        reader = csv.reader(self.file, delimiter=',')
        
        if self.K == 0:
            for _, line in enumerate(reader):
                i, j = int(line[0]), int(line[1])
                if self.v2p[i] == self.v2p[j]:
                    partition_id = self.v2p[j]
                else:
                    partition_id = self.v2p[i] if i < j else self.v2p[j]

                writer = csv.writer(file_objects[partition_id], delimiter=' ')
                writer.writerow(line)
            self.file.close()
        
        elif self.K == 1:
            for _, line in enumerate(reader):
                i, j = int(line[0]), int(line[1])
                partition_id = self.v2p[j]
                writer = csv.writer(file_objects[partition_id], delimiter=' ')
                writer.writerow(line)
            self.file.close()
        
        else:
            for l in range(self.number_partition):
                node_set = [k for k,v in self.v2p.items() if v == l]
                new_node_set = set()
                for _ in range(self.K):
                    self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
                    reader = csv.reader(self.file, delimiter=',')
                    for _, line in enumerate(reader):
                        i, j = int(line[0]), int(line[1])
                        if j in node_set:
                            writer = csv.writer(file_objects[l], delimiter=' ')
                            writer.writerow(line)
                            new_node_set.add(i)
                    node_set = new_node_set
                    self.file.close()
        
        
        for files in file_objects:
            files.close()
