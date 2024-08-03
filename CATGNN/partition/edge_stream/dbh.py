import numpy as np
import pandas as pd
import random
import json
import pickle
import os
import csv
import gzip
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from CATGNN.partition.partitioner import Partitioner
import hashlib


class DBH(Partitioner):
    """
    An implementation of Degree-based Hashing (DBH) partitioning method.
    For details about the algorithm see this paper:
    "Distributed Power-Law Graph Computing: Theoretical and Empirical Analysis"

    Args:
        dataset (str): Dataset name.
        path (str): Dataset root path.
        output_path (str): Output path.
        number_partition (int): Number of partitions.
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

        self.partition_features_file = partition_features_file
        self.print_partition_statistics = print_partition_statistics


    def partition(self):
        """Partition a graph."""
        self.get_degree()
        self.v2p = defaultdict(int)
        
        for i in range(self.number_partition):
            with open(os.path.join(self.output_path, 'partition_' + str(i) + '.txt'), 'w') as f:
                pass
        
        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
          
        file_objects = []
        for i in range(self.number_partition):
            filename = os.path.join(self.output_path, 'partition_' + str(i) + '.txt')
            files = open(filename, 'a', newline='')
            # Append the file object to the list
            file_objects.append(files)
        
        reader = csv.reader(self.file, delimiter=',')
        
        for _, line in enumerate(reader):
            i, j = int(line[0]), int(line[1])
            if self.node_degree[int(i)] < self.node_degree[int(j)]:
                hash_val = int(hashlib.sha256(str(i).encode()).hexdigest(), 16)
                partition_id = hash_val % self.number_partition
            else:
                hash_val = int(hashlib.sha256(str(j).encode()).hexdigest(), 16)
                partition_id = hash_val % self.number_partition
                
            self.v2p[j] = partition_id
            
            writer = csv.writer(file_objects[partition_id], delimiter=' ')
            writer.writerow(line)
        
        with open(self.output_path + 'partition' + '.json', 'wb') as json_file:
            pickle.dump(self.v2p, json_file)
        

        if self.K == 0:
            pass
        
        elif self.K == 1:
            self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
            reader = csv.reader(self.file, delimiter=',')
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
        