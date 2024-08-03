import os
import numpy as np
import pandas as pd
import json
import pickle
import csv
import dgl
import torch
import random
random.seed(42)
np.random.seed(42)
from CATGNN.utils import utils

"""
Data preprocessing functions used in CATGNN.    
"""


def get_subgraph(node_ids, 
                 K, 
                 dataset, 
                 name, 
                 path, 
                 output_path, 
                 save_dgl_graph):
    """
    Get the K-hops neighbor subgraph.
    
    Args:
        node_ids (list): List of node ID. 
        K (int): Number of hops neighbors.
        dataset (str): Datasest.
        name (str): Name of output.
        path (str): Input path.
        output_path (str): Output path.
        save_dgl_graph (str): Save partitioned graph as dgl graph object if Ture.
    """
    
    node_set = set(node_ids)
    new_node_set = set()
    for _ in range(K):
        with open(path + dataset + '/edge_list.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            out_file = open(output_path + dataset + '-' + name + '.txt', 'a')
            for _, line in enumerate(reader):
                i, j = int(line[0]), int(line[1])
                if j in node_set:
                    writer = csv.writer(out_file, delimiter=',')
                    writer.writerow(line)
                    new_node_set.add(i)
            out_file.close()
        file.close()
        node_set = new_node_set
        
    if save_dgl_graph:    
        edge_list = pd.read_csv(output_path + dataset + '-' + name + '.csv', sep=',', names=['src','dst'])
        edge_list, node_mapping = utils.pd_reindex_id(edge_list)

        with open(output_path + dataset + '-' + name + '-mapping.json', 'wb') as json_file:
            pickle.dump(node_mapping, json_file)

        g = dgl.graph((torch.tensor(edge_list['src']), torch.tensor(edge_list['dst'])))
        g = dgl.to_bidirected(g, copy_ndata=True)
        
        feat = np.load(output_path + dataset + '-' + name + '-feats.npy')
        label = np.load(output_path + dataset + '-' + name + '-labels.npy')
        g.ndata['feat'] = torch.from_numpy(feat).float()
        g.ndata['label'] = torch.from_numpy(label).to(torch.int64)
        
        # train_node_p = []
        if name == 'val':
            val_node_p = [node_mapping.get(item,item) for item in node_ids if item in node_mapping]
            vest_node_p = []
        if name == 'test':
            val_node_p = []
            test_node_p = [node_mapping.get(item,item) for item in node_ids if item in node_mapping]
        
        train_mask_p = torch.zeros(g.num_nodes(), dtype=torch.bool)
        test_mask_p = torch.zeros(g.num_nodes(), dtype=torch.bool)
        val_mask_p = torch.zeros(g.num_nodes(), dtype=torch.bool)

        for j in val_node_p:
            val_mask_p[j] = True
        
        for j in test_node_p:
            test_mask_p[j] = True
        
        g.ndata['train_mask'] = train_mask_p
        g.ndata['val_mask'] = val_mask_p
        g.ndata['test_mask'] = test_mask_p
        
        utils.save_graphs(output_path + dataset + '-' + name + '.bin', [g])


def random_features(dataset, 
                    n_nodes, 
                    feat_size, 
                    output_path):
    """
    Generate random node features
    Args:
        dataset (str): Dataset.
        n_nodes (int): Number of nodes.
        feat_size (int): Size (dimension) of features. 
        output_path (str): Output path.
    """
    
    isExist = os.path.exists(output_path + dataset)
    if not isExist:
        os.makedirs(dataset)
    
    features = np.random.random((n_nodes, feat_size)).astype(np.float32)
    np.save(output_path + dataset + '/feats.npy', features)
    return features


def random_labels(dataset, 
                  n_nodes, 
                  n_classes, 
                  output_path):
    """
    Generate random node labels
    Args:
        dataset (str): Dataset.
        n_nodes (int): Number of nodes.
        n_classes (int): Number of classes.
        output_path (str): Output path.
    """
    
    isExist = os.path.exists(output_path + dataset)
    if not isExist:
        os.makedirs(dataset)
    labels = np.random.randint(n_classes, size=n_nodes)
    labels_dict = dict()
    for i in range(len(labels)):
        labels_dict[str(i)] = int(labels[i])

    with open(output_path + dataset + '/class_map.json', 'w') as json_file:
        json.dump(labels_dict, json_file)
    
    return labels_dict


def train_valid_test_split(dataset, 
                           n_nodes, 
                           ratio, 
                           path):
    """
    Split the dataset into train/valid/test sets
    Args:
        dataset (str): Dataset.
        n_nodes (int): Number of nodes.
        ratio (list): List of split ratio.
        path (str): Dataset path.
    """
    
    isExist = os.path.exists(path + dataset)
    if not isExist:
        os.makedirs(dataset)
    nodes = [i for i in range(n_nodes)]
    random.shuffle(nodes)
    train_nodes = nodes[:int(ratio[0]*len(nodes))]
    test_nodes = nodes[int(ratio[0]*len(nodes)):int((ratio[0]+ratio[1])*len(nodes))]
    val_nodes = nodes[int((ratio[0]+ratio[1])*len(nodes)):]
    role = dict()
    role['tr'] = sorted(train_nodes)
    role['te'] = sorted(test_nodes)
    role['va'] = sorted(val_nodes)

    with open(path + dataset + '/role.json', 'w') as json_file:
        json.dump(role, json_file)
    
    return role
    

def process_dgl(dataset, path):
    """Process the DGL datasest"""
    
    isExist = os.path.exists(path + dataset)
    if not isExist:
        os.makedirs(dataset)
    
    if dataset == 'citeseer':
        data = dgl.data.CiteseerGraphDataset(raw_dir=path)
    elif dataset == 'pubmed':
        data = dgl.data.PubmedGraphDataset(raw_dir=path)
    elif dataset == 'cora':
        data = dgl.data.CoraGraphDataset(raw_dir=path)
    elif dataset == 'chameleon':
        data = dgl.data.ChameleonDataset(raw_dir=path)
    elif dataset == 'squirrel':
        data = dgl.data.SquirrelDataset(raw_dir=path)
    elif dataset == 'actor':
        data = dgl.data.ActorDataset(raw_dir=path)
    elif dataset == 'coauthor-cs':
        data = dgl.data.CoauthorCSDataset(raw_dir=path)
    elif dataset == 'coauthor-physics':
        data = dgl.data.CoauthorPhysicsDataset(raw_dir=path)
        
    graph = data[0]
    graph = dgl.to_bidirected(graph,copy_ndata=True)
    num_class = data.num_classes
    
    # get node feature
    features = graph.ndata['feat'].numpy()
    # print(np.shape(features))

    # get labels
    labels = graph.ndata['label']

    src_node = pd.DataFrame(graph.edges()[0].numpy())
    dst_node = pd.DataFrame(graph.edges()[1].numpy())
    # print(graph.edges())

    edge_list = pd.concat([src_node, dst_node], axis=1)
    np.savetxt(path + dataset + '/edge_list.txt', edge_list.values, fmt='%d')
    edge_list.to_csv(path + dataset + '/edge_list.csv', index=False, header=None)

    np.save(path + dataset + '/feats', features)

    if dataset in ['chameleon', 'squirrel', 'actor']:
        # get data split
        train_nodes = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0].tolist()
        test_nodes = torch.nonzero(graph.ndata['val_mask'], as_tuple=True)[0].tolist()
        val_nodes = torch.nonzero(graph.ndata['test_mask'], as_tuple=True)[0].tolist()
        # print(train_nodes)
    
    elif dataset in ['citeseer', 'pubmed', 'cora']:
        nodes = [i for i in range(graph.number_of_nodes())]
        test_nodes = torch.nonzero(graph.ndata['val_mask'], as_tuple=True)[0].tolist()
        val_nodes = torch.nonzero(graph.ndata['test_mask'], as_tuple=True)[0].tolist()

        # Create list d with elements in a but not in both b and c
        train_nodes = [x for x in nodes if x not in set(test_nodes) and x not in set(val_nodes)]
        
    else:
        nodes = [i for i in range(graph.number_of_nodes())]
        random.shuffle(nodes)
        train_nodes = nodes[:int(0.7*len(nodes))]
        test_nodes = nodes[int(0.7*len(nodes)):int(0.85*len(nodes))]
        val_nodes = nodes[int(0.85*len(nodes)):]
        
    role = dict()
    role['tr'] = sorted(train_nodes)
    role['te'] = sorted(test_nodes)
    role['va'] = sorted(val_nodes)


    with open(path + dataset + '/role.json', 'w') as json_file:
        json.dump(role, json_file)

    # labels_list = labels.tolist()
    labels_dict = dict()
    for i in range(len(labels)):
        labels_dict[str(i)] = int(labels[i])

    with open(path + dataset + '/class_map.json', 'w') as json_file:
        json.dump(labels_dict, json_file)


def create_dgl_graph(dataset, path, multilabel):
    """Create the DGL graph object""" 
    
    edge_list = pd.read_csv(path + dataset + '/edge_list.csv', header = None, names = ['src','dst'], index_col=False)

    src = edge_list['src'].to_list()
    dst = edge_list['dst'].to_list()

    graph = dgl.DGLGraph()

    graph.add_edges(src, dst)

    labels = json.load(open(path + dataset + '/class_map.json'))
    node_labels = np.array(list(labels.values()), dtype=np.int64)
    node_labels = torch.from_numpy(node_labels)
    graph.ndata['label'] = node_labels

    if multilabel:
        n_classes = len(graph.ndata['label'][0])
    else:
        n_classes = len(np.unique(node_labels))

    feats = np.load(path + dataset + '/feats.npy')
    node_features = torch.from_numpy(feats).float()
    graph.ndata['feat'] = node_features
    
    role = json.load(open(path + dataset +'/role.json'))

    train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    test_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    val_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)

    train_ids = role['tr']
    test_ids = role['te']
    val_ids = role['va']

    for i in train_ids:
        train_mask[i] = True

    for i in test_ids:
        test_mask[i] = True

    for i in val_ids:
        val_mask[i] = True

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    
    return graph, n_classes

    

