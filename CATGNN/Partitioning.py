import os
from CATGNN.partition import Hashing, DBH, Clustering, Greedy, HDRF, TwoPSL, SPRING, CustomPartitioner
from CATGNN.utils import utils
from CATGNN.utils import info
import warnings
warnings.filterwarnings('ignore')


class Partitioning(object):
    """
    Perform partitioning with the selected streaming partitioning algorithms.
    
    Currently support streaming partitioning algorithms:
        Hashing: Random Hashing algorithm.
        DBH: Degree-based Hashing algorithm from "Distributed Power-Law Graph Computing: Theoretical and Empirical Analysis".
        Greedy: Greedy Vertex-cut partitioning algorithm from "Powergraph: Distributed Graph-Parallel Computation on Natural Graphs".
        HDRF: High Degree (are) Replicated First (HDRF) partitioning algorithm from "HDRF: Stream-based Partitioning for Power-Law Graphs".
        2PSL: 2-Phase Streaming in Linear Runtime (2PSL) partitioning algorithm from "Out-of-core edge partitioning at linear run-time".
        Clustering: Clustering algorithm from  "A streaming algorithm for graph clustering".
        SPRING: Our new algorithms based on cluster-merge.
        Custom: A custom module supports any user-defined streaming partitioning algorithms.
    
    Args:
        dataset (str): Dataset name.
        multilabel (bool): whether the datatset is for multilabel classification.
        number_partition (int): Number of partitions. If 'auto', calculates the number of partitions based on system specs.
        seed (int): Random seed. Default is 42.
        path (str): Path of dataset directory.
        method (str): Partitioning method.
        output_path (str): Path of output directory.
        partition_features_file (bool): Partition the features file if True.
        print_partition_statistics (bool): Print out the statistics of the partitioned graph if True.
        save_dgl_graph (bool): Save partitioned graph as dgl graph object if Ture else in txt file.
        T (float): Memory needed for GNN training. Default is 'None', which means T is 2/3 of the total available GPU mempry.
        K (int): Number of hops of neighbor maintained after partitioning. Default is 1.
    """

    def __init__(self, 
                dataset: str = None,
                multilabel: bool=False,
                number_partition: int = 2,
                seed: int = 42,
                path: str = None,
                method: str = None,
                output_path: str = None, 
                partition_features_file: bool = True,
                print_partition_statistics: bool = True,
                save_dgl_graph: bool = True,
                T: float = None,
                K: int = 1):
        
        self.dataset = dataset
        self.multilabel = multilabel
        self.path = path
        self.output_path = output_path
        self.number_partition = number_partition
        self.T = T
        self.K = K
        
        isExist = os.path.exists(self.output_path)
        if not isExist:
            os.makedirs(self.output_path)
        
        self.seed = seed
        self.method = method
        
        self.partition_features_file = partition_features_file
        self.print_partition_statistics = print_partition_statistics
        self.save_dgl_graph = save_dgl_graph

        if self.number_partition == 'Auto':
            self.number_partition = info.auto_number_of_partition(self.dataset, 
                                                                  self.path, 
                                                                  self.T)
            
            if self.number_partition == 1:
                print('Graph will not be paritioned!')
                pass
            
            elif self.number_partition < 1:
                raise NotImplementedError('number_partition should equal to or greater than 1')
                

        if self.method == 'Clustering':
            self.sp = Clustering(dataset = self.dataset,
                                 multilabel = self.multilabel,
                                 path = self.path,
                                 output_path = self.output_path,
                                 number_partition = self.number_partition,
                                 K = self.K,
                                 stream_iters=1,
                                 seed = self.seed,
                                 partition_features_file = self.partition_features_file,
                                 print_partition_statistics = self.print_partition_statistics)
          
        elif self.method == 'SPRING':
            self.sp = SPRING(dataset = self.dataset, 
                             multilabel = self.multilabel, 
                             path = self.path, 
                             output_path = self.output_path,
                             number_partition = self.number_partition,
                             K = self.K,
                             stream_iters=1,
                             seed = self.seed,
                             partition_features_file = self.partition_features_file, 
                             print_partition_statistics = self.print_partition_statistics)
           
        elif self.method == 'Random':
            self.sp = Hashing(dataset = self.dataset, 
                              multilabel= self.multilabel, 
                              path = self.path, 
                              output_path = self.output_path,
                              number_partition = self.number_partition,
                              K = self.K,
                              seed = self.seed,
                              partition_features_file = self.partition_features_file,
                              print_partition_statistics = self.print_partition_statistics)
        
        elif self.method == 'DBH':
            self.sp = DBH(dataset = self.dataset, 
                          multilabel= self.multilabel, 
                          path = self.path, 
                          output_path = self.output_path,
                          number_partition = self.number_partition,
                          K = self.K,
                          seed = self.seed,
                          partition_features_file = self.partition_features_file,
                          print_partition_statistics = self.print_partition_statistics)
            
        elif self.method == 'Greedy':
            self.sp = Greedy(dataset = self.dataset, 
                             multilabel= self.multilabel, 
                             path = self.path, 
                             output_path = self.output_path,
                             number_partition = self.number_partition,
                             K = self.K, 
                             seed = self.seed,
                             partition_features_file = self.partition_features_file,
                             print_partition_statistics = self.print_partition_statistics)
          
        elif self.method == 'HDRF':
            self.sp = HDRF(dataset = self.dataset, 
                           multilabel= self.multilabel, 
                           path = self.path, 
                           output_path = self.output_path,
                           number_partition = self.number_partition,
                           K = self.K, 
                           Lambda = 1.0, 
                           seed = self.seed,
                           partition_features_file = self.partition_features_file,
                           print_partition_statistics = self.print_partition_statistics)
          
        elif self.method == '2PSL':
            self.sp = TwoPSL(dataset = self.dataset, 
                             multilabel= self.multilabel, 
                             path = self.path, 
                             output_path = self.output_path,
                             number_partition = self.number_partition,
                             K = self.K,
                             stream_iters =1, 
                             score='linear', 
                             eval_cluster=False, 
                             seed = self.seed,
                             partition_features_file = self.partition_features_file,
                             print_partition_statistics = self.print_partition_statistics)

        elif self.method == 'custom':
            self.sp = CustomPartitioner(dataset = self.dataset, 
                                        multilabel= self.multilabel, 
                                        path = self.path, 
                                        output_path = self.output_path,
                                        number_partition = self.number_partition,
                                        K = self.K,
                                        seed = self.seed,
                                        partition_features_file = self.partition_features_file,
                                        print_partition_statistics = self.print_partition_statistics)

        elif self.method == None:
            print('No paritition method is selected.')
        
        else:
            raise NotImplementedError('No Support for \'{}\' Yet. Please Try Different Methods.'.format(self.method))

    
    def run(self):
        if self.number_partition == 1 or self.method == None:
            print('Graph will not be partitioned!')
        
        else:
            self.sp.run()
            
            if self.save_dgl_graph:
                utils.save_dgl_graph(self.dataset, 
                                     self.path, 
                                     self.output_path, 
                                     self.number_partition)
            print("Partitioning Done!")
            print('='*60)