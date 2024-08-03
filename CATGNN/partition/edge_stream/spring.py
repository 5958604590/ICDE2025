import numpy as np
import pandas as pd
import random
import json
import pickle
import os
import csv
import gzip
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')
from CATGNN.partition.partitioner import Partitioner
from sortedcontainers import SortedDict


class SortedDictByValue:
    """An implementation of SortedDictByValue data structure."""
    
    def __init__(self):
        self.sorted_dict = SortedDict()
        self.keys_by_value = []

    def __setitem__(self, key, value):
        # Remove the old key if it exists
        if key in self.sorted_dict:
            self.__delitem__(key)

        # Add the key to the SortedDict
        self.sorted_dict[key] = value

        # Add the key to the list, maintaining the order based on values
        index = self._find_insert_index(key, value)
        self.keys_by_value.insert(index, key)

    def __delitem__(self, key):
        # Remove the key from the list
        self.keys_by_value.remove(key)

        # Remove the key from the SortedDict
        del self.sorted_dict[key]

    def _find_insert_index(self, key, value):
        # Find the index to insert the key based on values
        lo, hi = 0, len(self.keys_by_value)
        while lo < hi:
            mid = (lo + hi) // 2
            mid_key = self.keys_by_value[mid]
            mid_value = self.sorted_dict[mid_key]
            if mid_value < value or (mid_value == value and mid_key < key):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def items(self):
        # Yield items in sorted order based on values (and keys for equal values)
        for key in self.keys_by_value:
            yield key, self.sorted_dict[key]

    def get_ith_element(self, i):
        # Check if the index is within bounds
        if 0 <= i < len(self):
            # Use indexing to get the i-th element
            key = self.keys_by_value[i]
            return key, self.sorted_dict[key]
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        # Get the length based on the length of keys_by_value
        return len(self.keys_by_value)

    def __str__(self):
        # String representation of the sorted dictionary
        return str(dict(self.items()))



class SPRING(Partitioner):
    """An implementation of SPRING method.
    For details about the algorithm see our paper.

    Args:
        dataset (str): Dataset name.
        multilabel (bool): whether the datatset is for multilabel classification.
        path (str): Dataset root path.
        output_path (str): Output path.
        number_partition (int): Number of partitions.
        K (int): Number of hops of neighbor maintained after partitioning. Default is 1.
        stream_iters (int): Number of streaming iterations for clustering.
        seed (int): Random seed. Default is 42.
        partition_features_file (bool): Partition the features file if True.
        print_partition_statistics (bool): Print out the statistics of the partitioned graph if True.
    """

    def __init__(self, 
                 dataset: str = None, 
                 multilabel: bool=False, 
                 path: str = None, 
                 output_path: str = None, 
                 number_partition: int = 4,
                 K: int = 1, 
                 stream_iters: int = 1, 
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
        
        self.max_degree_neighbor = defaultdict(int)

 
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
                
                if int(j) not in self.max_degree_neighbor.keys():
                    self.max_degree_neighbor[int(j)] = int(i)
                
                elif self.node_degree[self.max_degree_neighbor[int(j)]] < self.node_degree[int(i)]:
                    self.max_degree_neighbor[int(j)] = int(i)
            

    def cluster_merge(self):

        cluster_nodes_dict = defaultdict(set)

        for node, cluster in self.v2c.items():
            cluster_nodes_dict[cluster].add(node)

        # Calculate the target size based on your condition
        target_size = 1.05 * self.number_nodes / self.number_partition

        # Create a dictionary to store cluster sizes
        cluster_sizes = Counter(self.v2c.values())
        # print('size_clusters: ', cluster_sizes)
        
        # Sort the clusters by size in ascending order
        sorted_clusters = SortedDictByValue()
        for cluster, size in cluster_sizes.items():
            sorted_clusters[cluster] = size


        n_loop = len(sorted_clusters) - 1
        for i in range(n_loop):
            smallest_cluster, size_smallest_cluster = sorted_clusters.get_ith_element(i)
            # print('smallest_cluster', smallest_cluster)
            # print('size_smallest_cluster', size_smallest_cluster)

            # Find nodes in the smallest cluster
            nodes_in_smallest_cluster = cluster_nodes_dict[smallest_cluster]
            # print('nodes_in_smallest_cluster: ', nodes_in_smallest_cluster)

            # Get the richest neighbor for each node in the smallest cluster
            richest_neighbors = [self.max_degree_neighbor[node] for node in nodes_in_smallest_cluster]
            richest_neighbors = list(set(richest_neighbors))  # Deduplicate neighbors
            # print('richest_neighbors: ', richest_neighbors)

            # Calculate the degree of each neighbor
            neighbor_degrees = {neighbor: self.node_degree[neighbor] for neighbor in richest_neighbors}
            # print('sub_dict: ')
            # pprint.pprint(neighbor_degrees)

            # Find the neighbor with the highest degree
            richest_of_richest_neighbors = max(neighbor_degrees, key=neighbor_degrees.get)
            richest_of_richest_neighbors_cluster = self.v2c[richest_of_richest_neighbors]
            # print('richest_of_richest_neighbors: ', richest_of_richest_neighbors)
            # print('richest_of_richest_neighbors_cluster: ', richest_of_richest_neighbors_cluster)

            if richest_of_richest_neighbors_cluster != smallest_cluster:
                # if richest_of_richest_neighbors_cluster in sorted_clusters.sorted_dict:
                if sorted_clusters.sorted_dict[richest_of_richest_neighbors_cluster] + size_smallest_cluster <= target_size:
                    # Merge the smallest cluster into the richest neighbor's cluster
                    for node in nodes_in_smallest_cluster:
                        self.v2c[node] = richest_of_richest_neighbors_cluster
                    sorted_clusters.sorted_dict[richest_of_richest_neighbors_cluster] += size_smallest_cluster
                    sorted_clusters.sorted_dict[smallest_cluster] -= size_smallest_cluster

                    cluster_nodes_dict[richest_of_richest_neighbors_cluster].update(cluster_nodes_dict[smallest_cluster])
                    del cluster_nodes_dict[smallest_cluster]

    def cluster2partition(self):

        self.v2p = defaultdict(int)
        self.c2p = defaultdict(int)

        v2c_counts = Counter(self.v2c.values())

        # p_size = np.array([0 for i in range(self.number_partition)])
        p_size = [0 for i in range(self.number_partition)]

        for value, count in v2c_counts.items():
            index_min_p_size = np.argmin(p_size)
            p_size[index_min_p_size] += count
            self.c2p[value] = index_min_p_size

        for key, value in self.v2c.items():
            # if value in self.c2p:
            self.v2p[key] = self.c2p[value]

        return self.v2p
    
    
    def partition(self):
        """Partition a graph."""

        for i in range(self.number_partition):
            with open(os.path.join(self.output_path, 'partition_' + str(i) + '.txt'), 'w') as f:
                pass
        
        self.get_degree()
        self.restream_clustering()
        self.cluster_merge()
        self.cluster2partition()
        # self.v2p = defaultdict(int)
        
        # for i in range(len(self.list_p)):
        #     for j in self.list_p[i]:
        #         self.v2p[j] = i
        
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
