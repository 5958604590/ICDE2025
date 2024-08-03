import numpy as np
import pandas as pd
import random
import json
import pickle
import os
import csv
from collections import defaultdict
from CATGNN.partition.partitioner import Partitioner
from CATGNN.utils import utils
import warnings
warnings.filterwarnings('ignore')

class CustomPartitioner(Partitioner):
    """
    The implementation of the custom partitioner module.

    Args:
        dataset (str): Dataset name.
        path (str): Dataset root path.
        output_path (str): Output path.
        number_partition (int): Number of partitions. Default is auto.
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

    
    def partition(self):
        self.node_degree = defaultdict(int)
        self.get_degree()
        self.v2p = defaultdict(int)
        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
        reader = csv.reader(self.file, delimiter=',')
        
        for _, line in enumerate(reader):
            i, j = int(line[0]), int(line[1])
            ### Implement the user-defined algorithms here

        
        
        self.file.close()
        utils.partition_file(self.dataset, self.path, self.output_path, self.v2p)
        
        with open(self.output_path + 'partition' + '.json', 'wb') as json_file:
            pickle.dump(self.v2p, json_file)
    
    
    """
    An example of Degree-based Hashing (DBH) partitioning method using the custom module.
 
    def partition(self):
        self.node_degree = defaultdict(int)
        self.get_degree()
        self.v2p = defaultdict(int)
        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
        reader = csv.reader(self.file, delimiter=',')
        
        for _, line in enumerate(reader):
            i, j = int(line[0]), int(line[1])
            
            if self.node_degree[int(i)] < self.node_degree[int(j)]:
                hash_val = utils.hash_function(i)
                partition_id = hash_val % self.number_partition
            
            else:
                hash_val = utils.hash_function(j)
                partition_id = hash_val % self.number_partition  
            
            self.v2p[j] = partition_id
        
        self.file.close()
        utils.partition_file(self.dataset, self.path, self.output_path, self.v2p)
        
        with open(self.output_path + 'partition' + '.json', 'wb') as json_file:
            pickle.dump(self.v2p, json_file)
    """

    