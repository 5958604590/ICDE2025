import numpy as np
import pandas as pd
import os
import json
import pickle
import torch
from collections import OrderedDict
import hashlib
import csv
import dgl
from dgl.data.utils import save_graphs, load_graphs
import torch.nn.functional as F

"""
Additional functions used in CATGNN.    
"""


def np_reindex_id(edge_list):
    """Re-index the node ID of a graph using numpy."""
    node_set = np.unique(edge_list)
    new_node_set = np.arange(len(node_set))
    node_mapping = dict(zip(node_set, new_node_set))
    mapping_arr = np.zeros(node_set.max()+1,dtype=new_node_set.dtype)
    mapping_arr[node_set] = new_node_set
    new_edge_list = mapping_arr[edge_list]
    
    return new_edge_list, node_mapping


def pd_reindex_id(edge_list):
    """Re-index the node ID of a graph using pandas."""
    
    node_set = pd.unique(edge_list[['src', 'dst']].values.ravel())
    node_set.sort()
    new_node_set = np.arange(len(node_set))
    node_mapping = dict(zip(node_set, new_node_set))
    edge_list['src'] = edge_list['src'].map(node_mapping)
    edge_list['dst'] = edge_list['dst'].map(node_mapping)
    
    return edge_list, node_mapping
    

def save_dgl_graph(dataset, 
                   path, 
                   output_path, 
                   number_partition):
    """Save the partitioned graph as DGL graph object."""
    
    with open(output_path + 'partition' + '.json', 'rb') as fp:
        node_partition_dict = pickle.load(fp)
    
    role = json.load(open(path + dataset + '/role.json'))
    train_ids = role['tr']
    test_ids = role['te']
    val_ids = role['va']
        
    for i in range(number_partition):
        edge_list = pd.read_csv(output_path + 'partition_' + str(i) + '.txt', sep=' ', names=['src','dst'])
        edge_list, node_mapping = pd_reindex_id(edge_list)

        with open(output_path + 'partition_' + str(i) + '-mapping.json', 'wb') as json_file:
            pickle.dump(node_mapping, json_file)
            
        g_p = dgl.graph((torch.tensor(edge_list['src']), torch.tensor(edge_list['dst'])))
        g_p = dgl.to_bidirected(g_p,copy_ndata=True)
        feat = np.load(output_path + 'partition_' + str(i) + '-feats.npy')
        label = np.load(output_path + 'partition_' + str(i) + '-labels.npy')
        g_p.ndata['feat'] = torch.from_numpy(feat).float()
        g_p.ndata['label'] = torch.from_numpy(label).to(torch.int64)
        
        train_node_p = [j for j in train_ids if node_partition_dict[j]==i]
        val_node_p = [j for j in val_ids if node_partition_dict[j]==i]
        test_node_p = [j for j in test_ids if node_partition_dict[j]==i]
        
        train_node_p = [node_mapping.get(item,item) for item in train_node_p if item in node_mapping]
        val_node_p = [node_mapping.get(item,item) for item in val_node_p if item in node_mapping]
        test_node_p = [node_mapping.get(item,item) for item in test_node_p if item in node_mapping]
        
        train_mask_p = torch.zeros(g_p.num_nodes(), dtype=torch.bool)
        test_mask_p = torch.zeros(g_p.num_nodes(), dtype=torch.bool)
        val_mask_p = torch.zeros(g_p.num_nodes(), dtype=torch.bool)

        for j in train_node_p:
            train_mask_p[j] = True

        for j in val_node_p:
            val_mask_p[j] = True

        for j in test_node_p:
            test_mask_p[j] = True
          
        g_p.ndata['train_mask'] = train_mask_p
        g_p.ndata['val_mask'] = val_mask_p
        g_p.ndata['test_mask'] = test_mask_p
        
        save_graphs(output_path + 'partition_' + str(i) + '.bin', [g_p])


def load_dgl_graph(dataset, output_path, i):
    """Load the DGL graph object."""
     
    glist, _ = load_graphs(output_path + 'partition_' + str(i) + '.bin', [0])
    
    return glist


def averaging_models(models, 
                     weights):
    """Averaging the model weights."""
    
    worker_state_dict = [x.state_dict() for x in models]
    weight_keys = list(worker_state_dict[0].keys())
    avg_state_dict = OrderedDict()
    
    for key in weight_keys:
        key_sum = 0
        
        for i in range(len(models)):
            key_sum = key_sum + weights[i] * worker_state_dict[i][key]
        avg_state_dict[key] = key_sum

    return avg_state_dict


def partition_features(dataset, 
                       path, 
                       output_path, 
                       number_partition, 
                       multilabel):
    """Partition the features of a graph."""
    
    node_feats = np.load(path + dataset +'/feats.npy', mmap_mode='r')
    labels = json.load(open(path + dataset + '/class_map.json'))
    node_labels = np.array(list(labels.values()), dtype=np.int64)

    if multilabel:
        n_classes = len(node_labels[0])
    else:
        n_classes = len(np.unique(node_labels))
    
    with open(output_path + 'num_classes.txt', 'w') as f:
        f.write(str(n_classes))

    n_feats = node_feats.shape[1]
    
    with open(output_path + 'num_feats.txt', 'w') as f:
        f.write(str(n_feats))
    
    for i in range(number_partition):
        edge_list = np.loadtxt(output_path + 'partition_' + str(i) + '.txt', dtype=int)
        node_set = np.unique(edge_list)
        feat = node_feats[node_set]
        np.save(output_path + 'partition_' + str(i) + '-feats.npy', feat)

        label = node_labels[node_set]
        np.save(output_path + 'partition_' + str(i) + '-labels.npy', label)


def save_csv(data, csv_file):
    """Save the graph in csv file."""
    
    with open(csv_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        for key, value in data.items():
            writer.writerow([key, value])


def hash_function(x):
    """A hash function."""
    
    return int(hashlib.sha256(str(x).encode()).hexdigest(), 16)


def partition_file(dataset, path, output_path, v2p):
    """Partition the graph file based on the partitioing results."""
    
    file = open(path + dataset + '/edge_list.csv', 'r')
    reader = csv.reader(file, delimiter=',')
    for number_edges, line in enumerate(reader):
        i, j = int(line[0]), int(line[1])
        partition_id = v2p[j]
        
        with open(os.path.join(output_path, 'partition_' + str(partition_id) + '.txt'), 'a') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow(line)
        
    file.close()
    

def activation_funcation(activation):
    """Activation functions used in GNNs."""
    
    if activation == 'relu':
        return F.relu
    
    elif activation == 'sigmoid':
        return F.sigmoid
    
    elif activation == 'tanh':
        return F.tanh
    
    elif activation == 'leaky_relu':
        return F.leaky_relu
    
    elif activation == None:
        return activation
    
    else:
        raise NotImplementedError('No Support for \'{}\' Yet. Please Try Different Activation Functions.'.format(activation))