import os
import CATGNN

"""
An example to run CATGNN.

In this example, we partition the 'cora' dataset into 4 partitions using SPRING,
and train a 2-layer GraphSAGE model with a sampling size of [25, 10] using CATGNN.

Different partitioning algorithms can be used in addition to SPRING, such as DBH, Greedy(PowerGraph), HDRF, and 2PSL.
Different GNN models are available, including GCN, GAT, GraphSAGE, etc.
All the hyperparameters are tunable. 

Please refer to our paper for more details.
"""

if __name__ == "__main__":

    path = os.path.abspath(os.getcwd())

    dataset = 'cora'
    print('='*60)
    print('dataset: ', dataset)
    
    ##### Streaming Paritioning #####
    
    number_partition = 4
    method = 'SPRING'
    print('method: ', method)
    
    partitioner = CATGNN.Partitioning(dataset = dataset, 
                                    multilabel = False,
                                    number_partition = number_partition, 
                                    path = path + '/datasets/', 
                                    method = method, 
                                    output_path = path + '/output/' + 'partitions-' + str(number_partition) + '/' + dataset + '/' + method + '/', 
                                    partition_features_file = True,
                                    print_partition_statistics = True,
                                    save_dgl_graph = True)
    partitioner.run()
    
    ##### GNN Model Training #####
    gnn_model = CATGNN.GNN(dataset = dataset, multilabel = False,
                            path = path + '/datasets/',
                            output_path = path + '/output/' + 'partitions-' + str(number_partition) + '/' + dataset + '/' + method + '/',
                            number_partition = number_partition, 
                            model='GraphSAGE', 
                            n_hidden= 256, 
                            n_layers = 2, 
                            fanout = [25,10],
                            dropout = 0.0,
                            aggregator = 'mean', 
                            activation = 'relu',
                            epochs = 100, 
                            batch_size = 512, 
                            epochs_eval = 1, 
                            epochs_avg = 1,
                            optimizer = 'adam', 
                            lr = 0.01)
    gnn_model.run()
    

