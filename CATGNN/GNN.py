import numpy as np
import os
os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import sklearn.metrics
from torch.multiprocessing import Process
from dgl.data.utils import load_graphs
from CATGNN.model import GCN, GAT, GraphSAGE, GCN_Full, GAT_Full, GraphSAGE_Full
from CATGNN.model import GraphSAINT, GATv2, ClusterGCN, SGC, NGNN_GCN
from CATGNN.model import CustomGNN
from CATGNN.utils import utils
from CATGNN.data import preprocess
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


class GNN(object):
    """
    Perform GNN computation.
    
    Currently support GNN models:
        GCN: "Semi-Supervised Classification with Graph Convolutional Networks".
        GAT: "Graph Attention Networks".
        GraphSAGE: "Inductive Representation Learning on Large Graphs".
        GraphSAINT: "Graphsaint: Graph Sampling based Inductive Learning Method".
        ClusterGCN: "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks".
        GATv2: "How Attentive are Graph Attention Networks?". 
        SGC: "Simplifying Graph Convolutional Networks".
        NGNN_GCN: "Network In Graph Neural Network".
        Custom: A custom module supports user-defined GNNs.
    
    Args:
        dataset (str): Dataset name.
        path (str): Path of dataset directory.
        output_path (str): Path of output directory, i.e, the location of partitioned dataset.
        
        number_partition (int): Number of partitions.
        model (str): GNN model. 'GraphSAGE', 'GCN', 'GAT'. Defaults: 'GraphSAGE'.
        
        n_hidden (int): Dimension of hidden layers.
        n_layers (int): Number of hidden layers.
        fanout (list): Sampler fan-out.
        heads (list): Sizes of attention heads.
        dropout (float): Dropout rate.
        aggregator (str): Aggregator types.
        activation (str): 'relu', 'sigmoid', 'tanh', 'leaky_relu'.
        
        epochs (int): Number of training epochs. Defaults: 10.
        batch_size (int): Number of batch sizes. Defaults: 256.
        epochs_eval (int): Evaluate the performance every epochs_eval epochs.
        
        optimizer (str): Optimizer for model training. Defaults: 'adam'.
        lr (float): learning rate. Defaults: 1e-3.

    """
    
    def __init__(self, 
                 dataset: str = None, 
                 multilabel: bool = False,
                 path: str = None, 
                 output_path: str = None,
                 number_partition: int = 2, 
                 model: str = 'GraphSAGE', 
                 n_hidden: int = 32, 
                 n_layers: int = 2, 
                 fanout: list = [25,10], 
                 heads: list = [2,2,2],
                 dropout: float = 0.0,
                 aggregator: str = 'mean', 
                 activation: str = 'relu',
                 epochs: int = 10, 
                 batch_size: int = 256, 
                 epochs_eval: int = 1, 
                 epochs_avg: int = 1,
                 optimizer: str = 'adam', 
                 lr: float = 1e-3):
        
        self.dataset = dataset
        self.multilabel = multilabel
        self.path = path
        self.output_path = output_path
        
        if number_partition == 'auto':
            self.number_partition = len([f for f in os.listdir(self.output_path) if f.endswith('.bin') and os.path.isfile(os.path.join(self.output_path, f))])
        else:
            self.number_partition = number_partition
        
        self.number_device = torch.cuda.device_count()
        
        self.model = model
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.fanout = fanout
        self.heads = heads
        self.dropout = dropout
        
        if number_partition != 1:
            with open(self.output_path + 'num_classes.txt','r') as f:
                self.n_classes = int(f.read())
                
            with open(self.output_path + 'num_feats.txt','r') as f:
                self.in_feats = int(f.read())
        
        self.aggregator = aggregator
        
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        if self.batch_size == 0:
            self.model += '_Full'
        self.epochs_eval = epochs_eval
        self.epochs_avg = epochs_avg
        self.optimizer = optimizer
        self.lr = lr
        
    
    def data_loader(self, proc_id):
        """ Data Loader """
        device = proc_id
        if self.number_partition == 1:
            graph, self.n_classes = preprocess.create_dgl_graph(self.dataset, 
                                                               self.path, 
                                                               self.multilabel)
            self.in_feats = graph.ndata['feat'].shape[1]
        else:
            graph = utils.load_dgl_graph(self.dataset, 
                                         self.output_path, 
                                         proc_id)[0]
        
        if self.model in ['GCN', 'GCN_Full', 'GAT', 'GAT_Full', 'GATv2', 'ClusterGCN', 'SGC', 'NGNN']:
            graph = dgl.add_self_loop(graph)
            
        # graph = graph.to(device)

        train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
        
        # print("device: ", device)
        # print("number of training nodes: ", len(train_nids))
        self.number_training = len(train_nids)
        
            
        if self.model in ['GraphSAGE', 'GraphSAGE_Full']:
            sampler = dgl.dataloading.NeighborSampler(self.fanout)
            
        elif self.model in ['GCN', 'GCN_Full', 'GAT', 'GAT_Full', 'GATv2', 'SGC', 'NGNN']:
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        
        elif self.model == 'ClusterGCN':
            sampler = dgl.dataloading.ClusterGCNSampler(graph, 100)
            
        elif self.model == 'GraphSAINT':
            sampler = dgl.dataloading.SAINTSampler(mode='walk', 
                                                   budget=self.fanout)
        
        elif self.model == 'CustomGNN':
            sampler = dgl.dataloading.NeighborSampler(self.fanout)
            
        
        if self.batch_size == 0:
            return graph
        
        self.train_dataloader = dgl.dataloading.DataLoader(
            # graph.to('cpu'),
            # train_nids.to('cpu'),
            graph,
            train_nids,
            sampler,
            device=device,
            use_ddp=False,
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=False,  
            num_workers=0,  
        )
        
        
        ########################################
        ### testing data
        if device == 0:
        
            g, _ = preprocess.create_dgl_graph(self.dataset, self.path, self.multilabel)
            if self.model in ['GCN', 'GCN_Full', 'GAT', 'GAT_Full', 'GATv2', 'ClusterGCN', 'SGC', 'NGNN']:
                g = dgl.add_self_loop(g)
            # g = g.to(device)

            valid_nids = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
            test_nids = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
            
            ########################################
            
            sampler2 = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
            
            self.valid_dataloader = dgl.dataloading.DataLoader(
                # graph,
                g,
                valid_nids,
                sampler2,
                device=device,
                use_ddp=False,
                batch_size=64,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
            
            self.test_dataloader = dgl.dataloading.DataLoader(
                # graph,
                g,
                test_nids,
                sampler2,
                device=device,
                use_ddp=False,
                batch_size=64,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
    
    
    def model_initial(self):
        """ Model initialization """
        # device = proc_id
        if self.model == 'GCN':
            gnn_model = GCN(self.in_feats, 
                            self.n_hidden, 
                            self.n_layers, 
                            self.n_classes, 
                            self.dropout, 
                            self.activation)
        
        elif self.model == 'GCN_Full':
            gnn_model = GCN_Full(self.in_feats, 
                            self.n_hidden, 
                            self.n_layers, 
                            self.n_classes, 
                            self.dropout, 
                            self.activation)
        
        elif self.model == 'SGC':
            gnn_model = SGC(self.in_feats, 
                            self.n_hidden, 
                            self.n_layers, 
                            self.n_classes, 
                            self.dropout, 
                            self.activation)
        
        elif self.model == 'NGNN_GCN':
            gnn_model = NGNN_GCN(self.in_feats, 
                            self.n_hidden, 
                            self.n_layers, 
                            self.n_classes, 
                            self.dropout, 
                            self.activation)
        
        elif self.model == 'GAT':
            self.heads = self.heads
            self.feat_dropout = self.dropout
            self.attn_dropout = self.dropout
            self.negative_slope = 0.2
            self.residual = False
            gnn_model = GAT(self.in_feats, 
                            self.n_hidden, 
                            self.n_layers, 
                            self.n_classes, 
                            self.heads,
                            self.feat_dropout, 
                            self.attn_dropout, 
                            self.negative_slope, 
                            self.residual, 
                            self.activation)
        
        elif self.model == 'GAT_Full':
            self.heads = self.heads
            self.feat_dropout = self.dropout
            self.attn_dropout = self.dropout
            self.negative_slope = 0.2
            self.residual = False
            gnn_model = GAT_Full(self.in_feats, 
                            self.n_hidden, 
                            self.n_layers, 
                            self.n_classes, 
                            self.heads,
                            self.feat_dropout, 
                            self.attn_dropout, 
                            self.negative_slope, 
                            self.residual, 
                            self.activation)
        
        elif self.model == 'GATv2':
            self.heads = self.heads
            self.feat_dropout = self.dropout
            self.attn_dropout = self.dropout
            self.negative_slope = 0.2
            self.residual = False
            gnn_model = GATv2(self.in_feats, 
                              self.n_hidden, 
                              self.n_layers, 
                              self.n_classes, 
                              self.heads,
                              self.feat_dropout, 
                              self.attn_dropout, 
                              self.negative_slope, 
                              self.residual, 
                              self.activation)
        
        elif self.model == 'GraphSAGE':
            gnn_model = GraphSAGE(self.in_feats, 
                                  self.n_hidden, 
                                  self.n_layers, 
                                  self.n_classes, 
                                  self.dropout, 
                                  self.aggregator, 
                                  self.activation)
        
        elif self.model == 'GraphSAGE_Full':
            gnn_model = GraphSAGE_Full(self.in_feats, 
                                  self.n_hidden, 
                                  self.n_layers, 
                                  self.n_classes, 
                                  self.dropout, 
                                  self.aggregator, 
                                  self.activation)
        
        elif self.model == 'GraphSAINT':
            gnn_model = GraphSAINT(self.in_feats, 
                                   self.n_hidden, 
                                   self.n_layers, 
                                   self.n_classes, 
                                   self.dropout, 
                                   self.aggregator, 
                                   self.activation)
        
        elif self.model == 'ClusterGCN':
            gnn_model = ClusterGCN(self.in_feats, 
                                   self.n_hidden, 
                                   self.n_layers, 
                                   self.n_classes, 
                                   self.dropout, 
                                   self.activation)
        
        elif self.model == 'CustomGNN':
            gnn_model = GCN(self.in_feats, 
                            self.n_hidden, 
                            self.n_layers, 
                            self.n_classes, 
                            self.dropout, 
                            self.activation)
        
        else:
            raise NotImplementedError('No Support for \'{}\' Yet. Please Try Different Methods.'.format(self.model))
        
        
        if self.optimizer == 'sgd':
            opt = torch.optim.SGD(gnn_model.parameters(), lr=self.lr)
        
        elif self.optimizer == 'adam':
            opt = torch.optim.Adam(gnn_model.parameters(), lr=self.lr)
        
        else:
            print('No Support for \'{}\' Yet. Adam is used instead.'.format(self.optimizer))
            opt = torch.optim.Adam(gnn_model.parameters(), lr=self.lr)
        
        return gnn_model, opt
    
    
    def average_models(self, model, proc_id):
        """ Model averaging"""
        device = proc_id
        
        # Weights based on number of training samples
        number_training = torch.tensor(self.number_training).to(device)
        total_training = torch.tensor(number_training).to(device)
        torch.distributed.all_reduce(total_training, op=torch.distributed.ReduceOp.SUM)
        
        weights = number_training/total_training

        for key, value in model.state_dict().items():
            
            value *= weights

            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)

            model.state_dict()[key] = value

    
    def evaluation(self, model):
        valid_predictions = []
        valid_labels = []
        test_predictions = []
        test_labels = []
        
        with torch.no_grad():
            ## valid
            for input_nodes, output_nodes, mfgs in self.valid_dataloader:
                inputs = mfgs[0].srcdata["feat"]
                
                if self.multilabel:
                    valid_labels.append(mfgs[-1].dstdata["label"].float().cpu().numpy())
                    valid_preds = model(mfgs, inputs).cpu().numpy()
                    # print('valid_preds', valid_preds)
                    valid_preds[valid_preds > 0.5] = 1
                    valid_preds[valid_preds <= 0.5] = 0
                    valid_predictions.append(valid_preds)
                
                else:
                    valid_labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                    valid_predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                                  
            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            # print('valid_labels:', valid_labels)
            # print('valid_predictions', valid_predictions)
            valid_accuracy = sklearn.metrics.f1_score(valid_labels, valid_predictions, average='micro')
            
            
            ## test
            for input_nodes, output_nodes, mfgs in self.test_dataloader:
                inputs = mfgs[0].srcdata["feat"]
                
                if self.multilabel:
                    test_labels.append(mfgs[-1].dstdata["label"].float().cpu().numpy())
                    test_preds = model(mfgs, inputs).cpu().numpy()
                    test_preds[test_preds > 0.5] = 1
                    test_preds[test_preds <= 0.5] = 0
                    test_predictions.append(test_preds)
                
                else:
                    test_labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
                    test_predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())    
                
            test_predictions = np.concatenate(test_predictions)
            test_labels = np.concatenate(test_labels)
            test_accuracy = sklearn.metrics.f1_score(test_labels, test_predictions, average='micro')
            
            print("Validation Accuracy: {:.6f}, Testing Accuracy: {:.6f}".format(valid_accuracy, test_accuracy))
            if self.best_accuracy < valid_accuracy:
                self.best_accuracy = valid_accuracy
                torch.save(model.state_dict(), self.best_model_path)
        
        return valid_accuracy, test_accuracy
    
    
    def evaluation_saint(self, g, model, device):
        g_features = g.ndata["feat"]
        labels = g.ndata["label"]
        val_mask = g.ndata["val_mask"].cpu().numpy()
        test_mask = g.ndata["test_mask"].cpu().numpy()
        valid_labels = labels[val_mask].float().cpu().numpy()
        test_labels = labels[test_mask].float().cpu().numpy()
        
        with torch.no_grad():
            predictions = model.inference(g, device, 128, 0, 'cpu').cpu().numpy()
            valid_preds = predictions[val_mask]
            test_preds = predictions[test_mask]
            if self.multilabel: 
                valid_preds[valid_preds > 0.5] = 1
                valid_preds[valid_preds <= 0.5] = 0
                test_preds[test_preds > 0.5] = 1
                test_preds[test_preds <= 0.5] = 0
            else:
                valid_preds = valid_preds.argmax(1)
                test_preds = test_preds.argmax(1)
                                  
            valid_accuracy = sklearn.metrics.f1_score(valid_labels, valid_preds, average='micro')
            test_accuracy = sklearn.metrics.f1_score(test_labels, test_preds, average='micro')
            
            print("Validation Accuracy: {:.6f}, Testing Accuracy: {:.6f}".format(valid_accuracy, test_accuracy))
            if self.best_accuracy < valid_accuracy:
                self.best_accuracy = valid_accuracy
                torch.save(model.state_dict(), self.best_model_path)
        
        return valid_accuracy, test_accuracy
        
        
    def evaluation_full(self, g, model):
        g_features = g.ndata["feat"]
        labels = g.ndata["label"]
        val_mask = g.ndata["val_mask"].cpu().numpy()
        test_mask = g.ndata["test_mask"].cpu().numpy()
        valid_labels = labels[val_mask].float().cpu().numpy()
        test_labels = labels[test_mask].float().cpu().numpy()
        
        with torch.no_grad():
            predictions = model(g, g_features).cpu().numpy()
            valid_preds = predictions[val_mask]
            test_preds = predictions[test_mask]
            if self.multilabel: 
                valid_preds[valid_preds > 0.5] = 1
                valid_preds[valid_preds <= 0.5] = 0
                test_preds[test_preds > 0.5] = 1
                test_preds[test_preds <= 0.5] = 0
            else:
                valid_preds = valid_preds.argmax(1)
                test_preds = test_preds.argmax(1)
                                  
            valid_accuracy = sklearn.metrics.f1_score(valid_labels, valid_preds, average='micro')
            test_accuracy = sklearn.metrics.f1_score(test_labels, test_preds, average='micro')
            
            print("Validation Accuracy: {:.6f}, Testing Accuracy: {:.6f}".format(valid_accuracy, test_accuracy))
            if self.best_accuracy < valid_accuracy:
                self.best_accuracy = valid_accuracy
                torch.save(model.state_dict(), self.best_model_path)
        
        return valid_accuracy, test_accuracy


    def model_training_centralized(self, device):
        # if self.model == 'GAT':
        if self.batch_size == 0:
            train_graph =  self.data_loader(device)
            test_graph, _ = preprocess.create_dgl_graph(self.dataset, self.path, self.multilabel)
            test_graph = dgl.add_self_loop(test_graph)
            test_graph = test_graph.to(device)
            model, opt = self.model_initial()
            model = model.to(device)
            self.best_accuracy = 0
            self.best_model_path = self.output_path + self.model  + "-model-best.pt"
            
            if self.multilabel:
                loss_fcn = nn.BCEWithLogitsLoss()
                labels = train_graph.ndata["label"].float()
            else:
                loss_fcn = nn.CrossEntropyLoss()
                labels = train_graph.ndata["label"]
            
            features = train_graph.ndata["feat"]
            train_mask = train_graph.ndata["train_mask"]
            results = []
            for epoch in range(1, self.epochs+1):
                model.train()
                predictions = model(train_graph, features)
                loss = loss_fcn(predictions[train_mask], labels[train_mask])
                opt.zero_grad()
                loss.backward()
                opt.step()
                model.eval()
            
                if epoch % self.epochs_eval == 0:
                    print('Epoch: ', epoch)
                    valid_accuracy, test_accuracy = self.evaluation_gat(test_graph, model)
                    results.append([valid_accuracy, test_accuracy])
        
            best_result = max(results, key=lambda x: x[0])
            print('Best Accuracy: ', best_result)
        
        else:
            self.data_loader(device)
            model, opt = self.model_initial()
            model = model.to(device)
            self.best_accuracy = 0
            self.best_model_path = self.output_path + self.model  + "-model-best.pt"
            
            if self.multilabel:
                loss_fcn = nn.BCEWithLogitsLoss()
            else:
                loss_fcn = nn.CrossEntropyLoss()
            
            results = []
            for epoch in range(1, self.epochs+1):
                model.train()
                
                for step, (input_nodes, output_nodes, mfgs) in enumerate(self.train_dataloader):
                    inputs = mfgs[0].srcdata["feat"]
                    if self.multilabel:
                        labels = mfgs[-1].dstdata["label"].float()
                    else:
                        labels = mfgs[-1].dstdata["label"]
                    
                    opt.zero_grad()
                    predictions = model(mfgs, inputs)
                    loss = loss_fcn(predictions, labels)
                    loss.backward()
                    opt.step()
                model.eval()

                # Evaluate on only the first GPU.
                if epoch % self.epochs_eval == 0:
                    print('Epoch: ', epoch)
                    valid_accuracy, test_accuracy = self.evaluation(model)
                    results.append([valid_accuracy, test_accuracy])
            
            best_result = max(results, key=lambda x: x[0])
            print('Best Accuracy: ', best_result)
        
        
    
    def model_training(self, proc_id, devices):
        """ Model Training (# of partitions <= # of GPUs)"""
        
        if self.model == 'GraphSAINT':
            self.data_loader(proc_id)
            model, opt = self.model_initial()
            model = model.to(proc_id)

            self.best_accuracy = 0
            self.best_model_path = self.output_path + self.model  + "-model-best.pt"
            
            if self.multilabel:
                loss_fcn = nn.BCEWithLogitsLoss()
            else:
                loss_fcn = nn.CrossEntropyLoss()
            
            results = []
            
            for epoch in range(1, self.epochs+1):
                model.train()
                
                for sg in self.train_dataloader:
                    sg = sg.to(proc_id)
                    x = sg.ndata['feat']
                    if self.multilabel:
                        y = sg.ndata['label'].float()
                    else:
                        y = sg.ndata['label']
                    y = sg.ndata['label']
                    m = sg.ndata['train_mask'].bool()
                    
                    opt.zero_grad()
                    predictions = model(sg, x)
                    # print('predictions: ', predictions)
                    loss = loss_fcn(predictions[m], y[m])
                    loss.backward()
                    opt.step()

                # torch.distributed.barrier(device_ids=[proc_id])
                torch.distributed.barrier()
                self.average_models(model,proc_id)
                torch.distributed.barrier()
                # torch.distributed.barrier(device_ids=[proc_id])
                model.eval()
        
                # Evaluate on only the first GPU.
                if proc_id == 0:
                    results = []
                if (epoch % self.epochs_eval == 0 and proc_id == 0):
                # if epoch % self.epochs_eval == 0:
                    print('Epoch: ', epoch)
                    valid_accuracy, test_accuracy = self.evaluation(model)
                    results.append([valid_accuracy, test_accuracy])
            
            if proc_id == 0:
                best_result = max(results, key=lambda x: x[0])
                print('Best Accuracy: ', best_result)
        

        self.data_loader(proc_id)
        model, opt = self.model_initial()
        model = model.to(proc_id)

        self.best_accuracy = 0
        self.best_model_path = self.output_path + self.model  + "-model-best.pt"
        
        if self.multilabel:
            loss_fcn = nn.BCEWithLogitsLoss()
        else:
            loss_fcn = nn.CrossEntropyLoss()
        
        results = []
        
        total_round = int(self.epochs/self.epochs_avg)
        tic = time.time()
        for round in range(1, total_round + 1):
            for epoch in range(1, self.epochs_avg + 1):
                tic_epoch = time.time()
                model.train()
                
                for step, (input_nodes, output_nodes, mfgs) in enumerate(self.train_dataloader):
                    mfgs = [mfg.int().to(proc_id) for mfg in mfgs]
                    
                    inputs = mfgs[0].srcdata["feat"]
                    if self.multilabel:
                        labels = mfgs[-1].dstdata["label"].float()
                    else:
                        labels = mfgs[-1].dstdata["label"]
                    # print('labels: ', labels)
                    opt.zero_grad()
                    predictions = model(mfgs, inputs)
                    # print('predictions: ', predictions)
                    loss = loss_fcn(predictions, labels)
                    loss.backward()
                    opt.step()

                toc_epoch = time.time()
                print("Epoch Time(s): {:.4f}".format(toc_epoch - tic_epoch))
            torch.distributed.barrier(device_ids=[proc_id])
            # torch.distributed.barrier()
            # if epoch % self.epochs_avg == 0:
            tic_avg = time.time()
            self.average_models(model,proc_id)
            toc_avg = time.time()
            print("Model Averaging Time(s): {:.4f}".format(toc_avg - tic_avg))
            # torch.distributed.barrier()
            torch.distributed.barrier(device_ids=[proc_id])
            model.eval()
    
            # # Evaluate on only the first GPU.
            # if proc_id == 0:
            #     results = []
            if (epoch % self.epochs_eval == 0 and proc_id == 0):
            # if epoch % self.epochs_eval == 0:
                print('Epoch: ', (round-1)*self.epochs_avg+epoch)
                valid_accuracy, test_accuracy = self.evaluation(model)
                results.append([valid_accuracy, test_accuracy])
            torch.distributed.barrier(device_ids=[proc_id])

        toc = time.time()
        print("Toal Time(s): {:.4f}".format(toc - tic))
        
        if proc_id == 0:
            best_result = max(results, key=lambda x: x[0])
            print('Best Accuracy: ', best_result)

    
    def model_training2(self, proc_id, devices):
        """ (# of partitions > # of GPUs) """
        
        device = torch.device(proc_id)
        # print('device: ', device)
        
        if self.batch_size == 0:
            for work in self.assign_work_list[proc_id]:
                # print('work: ', work)
                device_model, opt = self.model_initial()
                device_model = device_model.to(device)
                ### load model
                state = torch.load(self.output_path + self.model  + '-model-' + str(work) + '.pt')
                # state = torch.load(self.output_path + self.model  + '-model-' + str(work) + '-epoch-'+ str(epoch-1) +'.pt')
                device_model.load_state_dict(state['state_dict'])
                opt.load_state_dict(state['optimizer'])
                
                ### load graph
                train_graph = utils.load_dgl_graph(self.dataset, self.output_path, work)[0]
                train_graph = dgl.add_self_loop(train_graph)
                train_graph = train_graph.to(device)

                train_nids = torch.nonzero(train_graph.ndata['train_mask'], as_tuple=True)[0].to(device)

                if self.multilabel:
                    loss_fcn = nn.BCEWithLogitsLoss()
                    labels = train_graph.ndata["label"].float()
                else:
                    loss_fcn = nn.CrossEntropyLoss()
                    labels = train_graph.ndata["label"]
                
                features = train_graph.ndata["feat"]
                # labels = train_graph.ndata["label"]
                train_mask = train_graph.ndata["train_mask"]
                for _ in range(1, 2):
                    device_model.train()
                    device_model.train()
                    predictions = device_model(train_graph, features)
                    loss = loss_fcn(predictions[train_mask], labels[train_mask])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                state = {'state_dict': device_model.state_dict(), 'optimizer': opt.state_dict()}
                # torch.save(state, self.output_path + self.model  + '-model-' + str(work) + '-epoch-'+ str(epoch) +'.pt')
                torch.save(state, self.output_path + self.model  + '-model-' + str(work) + '.pt')
                # torch.distributed.barrier(device_ids=[proc_id])
                
                del device_model
                del train_graph
                torch.cuda.empty_cache()
            torch.distributed.barrier(device_ids=[proc_id])
            torch.distributed.destroy_process_group()
            
        else:
            for work in self.assign_work_list[proc_id]:
                # print('work: ', work)
                device_model, opt = self.model_initial()
                device_model = device_model.to(device)
                ### load model
                state = torch.load(self.output_path + self.model  + '-model-' + str(work) + '.pt')
                # state = torch.load(self.output_path + self.model  + '-model-' + str(work) + '-epoch-'+ str(epoch-1) +'.pt')
                device_model.load_state_dict(state['state_dict'])
                opt.load_state_dict(state['optimizer'])
                
                ### load graph
                graph = utils.load_dgl_graph(self.dataset, self.output_path, work)[0]

                # train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0].to(device)
                train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
                # self.number_training = len(train_nids)


                if self.model in ['GraphSAGE', 'GraphSAGE_Full']:
                    sampler = dgl.dataloading.NeighborSampler(self.fanout)

                elif self.model in ['GCN', 'GCN_Full', 'GAT', 'GAT_Full', 'GATv2', 'SGC', 'NGNN']:
                    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
                
                elif self.model == 'ClusterGCN':
                    sampler = dgl.dataloading.ClusterGCNSampler(graph,100)
                    
                elif self.model == 'GraphSAINT':
                    sampler = dgl.dataloading.SAINTSampler(mode='walk', budget=self.fanout)
                    
                    
                
                self.train_dataloader = dgl.dataloading.DataLoader(
                    graph,
                    train_nids,
                    sampler,
                    device='cpu',
                    use_ddp=False,
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    drop_last=False,  
                    num_workers=0,  
                )
                
                if self.multilabel:
                    loss_fcn = nn.BCEWithLogitsLoss()
                else:
                    loss_fcn = nn.CrossEntropyLoss()
                
                # for epoch in range(1, self.epochs+1):
                    
                device_model.train()
                # for _ in range(10):
                for step, (input_nodes, output_nodes, mfgs) in enumerate(self.train_dataloader):
                    mfgs = [mfg.to(device) for mfg in mfgs]
                    inputs = mfgs[0].srcdata["feat"]
                    if self.multilabel:
                        labels = mfgs[-1].dstdata["label"].float()
                    else:
                        labels = mfgs[-1].dstdata["label"]
                    # print('labels: ', labels)
                    opt.zero_grad()
                    predictions = device_model(mfgs, inputs)
                    # print('predictions: ', predictions)
                    loss = loss_fcn(predictions, labels)
                    loss.backward()
                    opt.step()

                state = {'state_dict': device_model.state_dict(), 'optimizer': opt.state_dict()}
                # torch.save(state, self.output_path + self.model  + '-model-' + str(work) + '-epoch-'+ str(epoch) +'.pt')
                torch.save(state, self.output_path + self.model  + '-model-' + str(work) + '.pt')
                # torch.distributed.barrier(device_ids=[proc_id])
                
                del device_model
                del graph
                del self.train_dataloader
                torch.cuda.empty_cache()
            torch.distributed.barrier(device_ids=[proc_id])
            torch.distributed.destroy_process_group()
                

    def init_processes(self, rank, size, fn, backend='nccl'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        torch.distributed.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)
    
    
    def run(self):
        # print('=========='*5)
        print('Start GNN Training!')
        
        ## Centralized model training
        if self.number_partition == 1:
            print('Model is training in centralized manner!')
            # self.model_training_centralized(f'cuda:{torch.cuda.device_count()-1}')
            self.model_training_centralized(0)
            
        ## Distributed model training (# of partitions <= # of GPUs)
        elif self.number_partition <= self.number_device:
            processes = []
            torch.multiprocessing.set_start_method("spawn")
            # for rank in range(self.number_device):
            for rank in range(self.number_partition):
                # p = Process(target=self.init_processes, args=(rank, self.number_device, self.model_training))
                p = Process(target=self.init_processes, args=(rank, self.number_partition, self.model_training))
                
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        
        ## Distributed model training (# of partitions > # of GPUs)
        else:
            ### create work list for each device
            work_list = [i for i in range(self.number_partition)]
            self.assign_work_list = [[] for _ in range(self.number_device)]

            current_part = 0
            for work in work_list:
                self.assign_work_list[current_part].append(work)
                current_part = (current_part + 1) % self.number_device
            # print('assign_work_list: ', self.assign_work_list)
            
            ### initial models
            for i in range(self.number_partition):
                init_model, init_optimizer = self.model_initial()
                ## save model
                init_state = {'state_dict': init_model.state_dict(), 'optimizer': init_optimizer.state_dict()}
                torch.save(init_state, self.output_path + self.model  + '-model-' + str(i) + '.pt')
                # torch.save(init_state, self.output_path + self.model  + '-model-' + str(i) + '-epoch-0' +'.pt')
                
            global_model, global_optimizer = self.model_initial()
            
            # init_model, init_optimizer = self.model_initial()
            
            ### calculate model weights
            weights = []
            for i in range(self.number_partition):
                graph = utils.load_dgl_graph(self.dataset, self.output_path, i)[0]
                train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
                weights.append(len(train_nids))
            sum_weights = sum(weights)
            weights = [i/sum_weights for i in weights]
            # print('weights', weights)
            
            results = []
            self.best_accuracy = 0
            self.best_model_path = self.output_path + self.model  + "-model-best.pt"
            
            
            ### model training
            for epoch in range(1, self.epochs + 1):
                # train the model
                processes = []
                torch.multiprocessing.set_start_method('spawn', force=True)
                for rank in range(self.number_device):
                    p = Process(target=self.init_processes, args=(rank, self.number_device, self.model_training2))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
                
                processes.clear()

                # load the model and take average
                models, optimizers = [], []
                for i in range(self.number_partition):
                    
                    ## load model
                    local_model, local_optimizer = self.model_initial()
                    local_state = torch.load(self.output_path + self.model  + '-model-' + str(i) + '.pt')
                    # local_state = torch.load(self.output_path + self.model  + '-model-' + str(i) + '-epoch-'+ str(epoch) +'.pt')
                    
                    local_model.load_state_dict(local_state['state_dict'])
                    local_optimizer.load_state_dict(local_state['optimizer'])
                    models.append(local_model)
                    optimizers.append(local_optimizer)
                
                global_model, _ = self.model_initial()
                global_model_dict = utils.averaging_models(models, weights)
                global_model.load_state_dict(global_model_dict)
                torch.save(global_model_dict, self.output_path + self.model  + '-model-global' + '.pt')
                
                for i in range(self.number_partition):
                    # models[i].load_state_dict(final_model_dict)
                    ## save final_model to each model
                    state = {'state_dict': global_model_dict, 'optimizer': optimizers[i].state_dict()}
                    torch.save(state, self.output_path + self.model  + '-model-' + str(i) + '.pt')
                    # torch.save(state, self.output_path + self.model  + '-model-' + str(i) + '-epoch-'+ str(epoch) +'.pt')
                global_model.to('cuda')
                
                if epoch % self.epochs_eval == 0:
                
                    ### testing data
                    g, _ = preprocess.create_dgl_graph(self.dataset, self.path, self.multilabel)

                    valid_nids = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
                    test_nids = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
                    
                    sampler2 = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)

                    self.valid_dataloader = dgl.dataloading.DataLoader(
                        # graph,
                        g,
                        valid_nids,
                        sampler2,
                        device='cuda',
                        use_ddp=False,
                        batch_size=64,
                        shuffle=False,
                        drop_last=False,
                        num_workers=0,
                        # use_uva=True
                    )
                    
                    self.test_dataloader = dgl.dataloading.DataLoader(
                        # graph,
                        g,
                        test_nids,
                        sampler2,
                        device='cuda',
                        use_ddp=False,
                        batch_size=64,
                        shuffle=False,
                        drop_last=False,
                        num_workers=0,
                        # use_uva=True
                    )
                
                    
                    print('Epoch: ', epoch)
                    global_model.eval()
                    valid_accuracy, test_accuracy = self.evaluation(global_model)
                    results.append([valid_accuracy, test_accuracy])
                    
                    ## clean up GPU cache
                    del global_model
                    del g
                    del self.valid_dataloader
                    del self.test_dataloader
                    torch.cuda.empty_cache()
                    
            best_result = max(results, key=lambda x: x[0])
            
            print('Best Accuracy: ', best_result)
            print('=========='*5)
            