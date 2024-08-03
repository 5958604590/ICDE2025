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
# from memory_profiler import profile
from CATGNN.partition.partitioner import Partitioner
UINT64_MAX = 2147483647


class TwoPSL(Partitioner):
    """
    An implementation of 2-Phase Streaming in Linear Runtime (2PSL) partitioning method.
    For details about the algorithm see this paper:
    "Out-of-core edge partitioning at linear run-time"

    Args:
        dataset (str): Dataset name.
        number_partition (int): Number of partitions.
        stream_iters (int): Number of streaming iterations for clustering.
        score (str): Partitioning score used in the 2nd phase. 'linear' or 'hdrf'.
        eval_cluster (bool): Evaluate the clustering quality if True else False.
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
                 stream_iters: int =1, 
                 score: str='linear', 
                 eval_cluster: bool=False, 
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

        self.partition_features_file = partition_features_file
        self.print_partition_statistics = print_partition_statistics

        self.node_degree = defaultdict(int)
        self.stream_iters = stream_iters
        self.cluster_quality_eval = eval_cluster
        self.score = score

        self.max_num_partition = 256
        self.get_degree()
        ####################################### two phase parameters #######################################
        self.balance_ratio = 1.05
        self._lambda = 1
        self.max_vol = int(0.1*self.number_edges/self.number_partition)

        self.max_partition_load = self.balance_ratio*self.number_edges/self.number_partition
        # for streamcom
        self.volumes = [0 for _ in range(self.number_nodes+1)] #  index is community id, volume of a community
        self.communities = [0 for _ in range(self.number_nodes)] # index is vertex id, community of a vertex
        self.communities = defaultdict(int)
        self.quality_scores = [0.0 for _ in range(self.number_nodes+1)] # quality of the communities (intra-cluster edges / inter-cluster edges)
        self.next_community_id = 1

        # for partition
        self.edge_load = [0 for _ in range(self.number_partition)]
        self.vertex_partition_matrix = [ [False for _ in range(self.max_num_partition)] for _ in range(self.number_nodes)]

        self.partition_volume = [0 for _ in range(self.number_partition)]
        self.com2part = [0 for _ in range(self.number_nodes+1)]
        self.max_load = 0
        self.min_load = UINT64_MAX
        self.epsilon =1
        self.edge_p = defaultdict(list)


    def find_communities(self):
        self.do_streamcom()
        if self.cluster_quality_eval:
            self.evaluate_communities()
        
        self.do_streamcom()
        for i in range(3, self.stream_iters):
            if self.cluster_quality_eval:
                self.evaluate_communities()
            self.do_streamcom()


    def do_streamcom(self):
        
        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
        
        reader = csv.reader(self.file, delimiter=',')
        for _, edge in enumerate(reader):
            i, j = int(edge[0]), int(edge[1])
        
            if self.communities[i] == 0:
                self.communities[i] = self.next_community_id
                self.volumes[self.communities[i]] += self.node_degree[i]
                self.next_community_id += 1

            if self.communities[j] == 0:
                self.communities[j] = self.next_community_id
                self.volumes[self.communities[j]] += self.node_degree[j]
                self.next_community_id += 1

            real_vol_i = self.volumes[self.communities[i]]- self.node_degree[i]
            real_vol_j = self.volumes[self.communities[j]] - self.node_degree[j]


            if self.volumes[self.communities[i]] <= self.max_vol and self.volumes[self.communities[j]] <= self.max_vol:
                if self.cluster_quality_eval:
                    score_i = self.quality_scores[self.communities[i]]
                    score_j = self.quality_scores[self.communities[j]]
                    if real_vol_i <= real_vol_j and score_i >= score_j and self.volumes[self.communities[j]] + self.node_degree[i] <= self.max_vol:
                        self.volumes[self.communities[i]] -= self.node_degree[i]
                        self.volumes[self.communities[j]] += self.node_degree[i]
                        self.communities[i] = self.communities[j]
                    elif real_vol_j < real_vol_i and score_j >= score_i and self.volumes[self.communities[i]] + self.node_degree[j] <= self.max_vol:
                        self.volumes[self.communities[j]] -= self.node_degree[j]
                        self.volumes[self.communities[i]] += self.node_degree[j]
                        self.communities[j] = self.communities[i]

                else:
                    if real_vol_i <= real_vol_j and self.volumes[self.communities[j]] + self.node_degree[i] <= self.max_vol:
                        self.volumes[self.communities[i]] -= self.node_degree[i]
                        self.volumes[self.communities[j]] += self.node_degree[i]
                        self.communities[i] = self.communities[j]

                    elif real_vol_j < real_vol_i and self.volumes[self.communities[i]] + self.node_degree[j] <= self.max_vol:
                        self.volumes[self.communities[j]] -= self.node_degree[j]
                        self.volumes[self.communities[i]] += self.node_degree[j]
                        self.communities[j] = self.communities[i]


    def evaluate_communities(self):
        self.external_degree = defaultdict(int)
        
        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
        
        reader = csv.reader(self.file, delimiter=',')
        for _, edge in enumerate(reader):
            i, j = int(edge[0]), int(edge[1])

            if self.communities[i] != self.communities[j]:
                self.external_degree[self.communities[i]] += 1
                self.external_degree[self.communities[j]] += 1

        for i in range(self.number_nodes):
            demoniator = min(self.volumes[i], 2*self.number_edges - self.volumes[i])
            if demoniator !=0:
                score = self.external_degree[i] /demoniator
                self.quality_scores[i] = score

    def prepartition_and_partition(self):
        # phase 1
        sorted_communities = []
        for i in range(len(self.volumes)):
            sorted_communities.append((self.volumes[i], i))
        sorted_communities = sorted(sorted_communities,key=lambda x: x[0], reverse=True)

        for com_vol_pair in sorted_communities:
            if com_vol_pair[0]==0: break
            min_p = self.find_min_vol_partition()
            self.partition_volume[min_p] += com_vol_pair[0]
            self.com2part[com_vol_pair[1]] = min_p

        self.node_p_dict = defaultdict(int)
        for i in range(len(self.communities)):
            self.node_p_dict[i] = self.com2part[self.communities[i]]
            
        size_p = defaultdict(int)
        for k, v in self.node_p_dict.items():

            size_p[v] += 1

        # prepartition
        self.sort_com_prepartitioning()
        sum_edge_load = 0
        for i in range(len(self.edge_load)):
            sum_edge_load +=self.edge_load[i]

        # phase 2 begin here
        if self.score == 'hdrf':
            self.do_hdrf()
        elif self.score == 'linear':
            self.do_linear()


    def sort_com_prepartitioning(self):
        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
        
        reader = csv.reader(self.file, delimiter=',')
        for _, edge in enumerate(reader):
            i, j = int(edge[0]), int(edge[1])
            com_i = self.communities[i]
            com_j = self.communities[j]

            if com_i == com_j or self.com2part[com_i] == self.com2part[com_j]:
                partition = self.com2part[com_i]
                load = self.edge_load[partition]
                if load >= self.max_partition_load:
                    partition = self.find_max_score_partition(edge)

                self.update_vertex_partition_matrix(edge, partition)
                self.update_min_max_load(partition)
                self.v2p[j] = partition
                # self.v2p_set[j].add(partition)
                with open(os.path.join(self.output_path, 'partition_' + str(partition) + '.txt'), 'a') as f:
                    writer = csv.writer(f, delimiter=' ')
                    writer.writerow(edge)


    def find_max_score_partition_linear(self, edge):
        i, j = int(edge[0]), int(edge[1])

        degree_i = self.node_degree[i]
        degree_j = self.node_degree[j]
        max_score = 0
        max_p = 0

        if degree_i > degree_j:
            max_p = i%self.number_partition
        else:
            max_p = j%self.number_partition

        if self.edge_load[max_p] >= self.max_partition_load:
            min_load = -1
            min_p = 0
            for i in range(self.number_partition):
                if self.edge_load[i] < min_load:
                    min_load = self.edge_load[i]
                    min_p = i

            max_p = min_p
        return max_p

    def find_max_score_partition(self, edge):
        # i, j = edge.strip().split(' ')
        # i, j = int(i), int(j)
        i, j = int(edge[0]), int(edge[1])

        degree_i = self.node_degree[i]
        degree_j = self.node_degree[j]
        com_part_i = self.com2part[self.communities[i]]
        com_part_j = self.com2part[self.communities[j]]
        max_score = 0
        max_p = 0

        if self.edge_load[com_part_i] >= self.max_partition_load or self.edge_load[com_part_j] >= self.max_partition_load:
            if degree_i > degree_j:
                max_p = i%self.number_partition
            else:
                max_p = j%self.number_partition

            if self.edge_load[max_p] >= self.max_partition_load:
                min_load = -1
                min_p = 0
                for i in range(self.number_partition):
                    if self.edge_load[i] < min_load:
                        min_load = self.edge_load[i]
                        min_p = i

                max_p = min_p
            return max_p

        else:
            ps = [com_part_i, com_part_j]

            for p in ps:
                if self.edge_load[p] >= self.max_partition_load: continue

                gu = 0; gv = 0; gu_c = 0; gv_c = 0
                sum = degree_i + degree_j
                sum_of_volumes = self.volumes[self.communities[i]] + self.volumes[self.communities[j]]
                if self.vertex_partition_matrix[i][p]:
                    gu = degree_i
                    gu /= sum
                    gu = 1 + (1-gu)
                    if self.com2part[self.communities[i]] == p:
                        gu_c = self.volumes[self.communities[i]]
                        gu_c /= sum_of_volumes

                if self.vertex_partition_matrix[j][p]:
                    gv = degree_j
                    gv /= sum
                    gv = 1+(1-gv)
                    if self.com2part[self.communities[j]] == p:
                        gv_c = self.volumes[self.communities[j]]
                        gv_c /= sum_of_volumes

                score_p = gu+gv+gu_c+gv_c
                if score_p < 0:
                    print("ERROR score_p<0")
                    exit()
                if score_p >= max_score:
                    max_score = score_p
                    max_p = p

            return max_p


    def find_max_score_partition_hdrf(self, edge):
        i, j = int(edge[0]), int(edge[1])

        degree_i = self.node_degree[i]
        degree_j = self.node_degree[j]
        max_score = 0
        max_p = 0

        for p in range(self.number_partition):
            if self.edge_load[p]>= self.max_partition_load: continue

            gu=0; gv=0
            sum = degree_i+degree_j
            if self.vertex_partition_matrix[i][p]:
                gu=degree_i
                gu/=sum
                gu=1+(1-gu)

            if self.vertex_partition_matrix[j][p]:
                gv = degree_j
                gv/=sum
                gv = 1+(1-gv)

            bal = self.max_load - self.edge_load[p]
            if self.min_load !=UINT64_MAX:
                bal /= self.epsilon + self.max_load - self.min_load
            score_p = gu+gv+self._lambda*bal

            if score_p <0:
                print("ERROR: score_p <0")
                exit()
            if score_p > max_score:
                max_score = score_p
                max_p = p

        return max_p


    def update_vertex_partition_matrix(self, e, max_p):
        i, j = int(e[0]), int(e[1])
        
        self.vertex_partition_matrix[i][max_p] = True
        self.vertex_partition_matrix[j][max_p] = True


    def update_min_max_load(self, max_p):
        self.edge_load[max_p] += 1
        load = self.edge_load[max_p]
        if load > self.max_load:
            self.max_load = load


    def do_hdrf(self):
        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
        
        reader = csv.reader(self.file, delimiter=',')
        for _, edge in enumerate(reader):
            i, j = int(edge[0]), int(edge[1])

            com_i = self.communities[i]
            com_j = self.communities[j]
            if com_i == com_j or self.com2part[com_i]== self.com2part[com_j]:
                continue

            max_p = self.find_max_score_partition_hdrf(edge)
            self.update_vertex_partition_matrix(edge, max_p)
            self.update_min_max_load(max_p)

            self.v2p[j] = max_p
            with open(os.path.join(self.output_path, 'partition_' + str(max_p) + '.txt'), 'a') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerow(edge)


    def do_linear(self):
        
        self.file = open(self.path + self.dataset + '/edge_list.csv', 'r')
        
        reader = csv.reader(self.file, delimiter=',')
        for _, edge in enumerate(reader):
            i, j = int(edge[0]), int(edge[1])

            com_i = self.communities[i]
            com_j = self.communities[j]
            if com_i == com_j or self.com2part[com_i]== self.com2part[com_j]:
                continue
            max_p = self.find_max_score_partition_linear(edge)
            self.update_vertex_partition_matrix(edge, max_p)
            self.update_min_max_load(max_p)
            
            self.v2p[j] = max_p
            with open(os.path.join(self.output_path, 'partition_' + str(max_p) + '.txt'), 'a') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerow(edge)
            

    def find_min_vol_partition(self):
        min_p_vol = self.partition_volume[0]
        min_p = 0
        for i in range(1, self.number_partition):
            if self.partition_volume[i] < min_p_vol:
                min_p_vol = self.partition_volume[i]
                min_p = i
        return min_p


    def partition(self):
        """Partition a graph."""
        self.v2p = defaultdict(int)
        for i in range(self.number_partition):
            with open(os.path.join(self.output_path, 'partition_' + str(i) + '.txt'), 'w') as f:
                pass

        self.find_communities()
        self.prepartition_and_partition()
        
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