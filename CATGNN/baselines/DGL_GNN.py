import os
import sys
import numpy as np
import pandas as pd
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import sklearn.metrics
os.environ["DGLBACKEND"] = "pytorch"

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats: int = 32,
                 n_hidden: int = 32,
                 n_layers: int = 2,
                 n_classes: int = 32,
                 dropout: float = 0.0,
                 aggregator: str = 'mean',
                 activation: str = 'relu'):
        super().__init__()

        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.aggregator = aggregator

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == None:
            self.activation = activation
        else:
            raise NotImplementedError('No Support for \'{}\' Yet. Please Try Different Activation Functions.'.format(activation))

        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.SAGEConv(self.in_feats,
                                           self.n_hidden,
                                           self.aggregator,
                                           activation=self.activation))

        for i in range(self.n_layers-2):
            self.layers.append(dgl.nn.SAGEConv(self.n_hidden,
                                               self.n_hidden,
                                               self.aggregator,
                                               activation=self.activation))

        self.layers.append(dgl.nn.SAGEConv(self.n_hidden,
                                           self.n_classes,
                                           self.aggregator))

        self.dropout = nn.Dropout(dropout)


    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != 0:
                h = self.dropout(h)
        return h


    def inference(self,
                  g,
                  device,
                  batch_size,
                  num_workers,
                  buffer_device=None):

        feat = g.ndata['feat']

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

        dataloader = dgl.dataloading.DataLoader(g,
                                                torch.arange(g.num_nodes()).to(g.device),
                                                sampler,
                                                device=device,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=num_workers)

        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device, pin_memory=True)
            feat = feat.to(device)

            for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y

        return y

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


class GNN(object):
    def __init__(self, dataset: str = None, multilabel: bool = False,
        path: str = None, output_path: str = None,
        model: str = 'GraphSAGE',
        n_hidden: int = 32, n_layers: int = 2, fanout: list = [25,10], dropout: float = 0.0,
        aggregator: str = 'mean', activation: str = 'relu',
        epochs: int = 10, batch_size: int = 256, epochs_eval: int = 5,
        optimizer: str = 'adam', lr: float = 1e-3):

        self.dataset = dataset
        self.multilabel = multilabel
        self.path = path
        self.output_path = output_path

        isExist = os.path.exists(self.output_path)
        if not isExist:
            os.makedirs(self.output_path)

        self.model = model

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.fanout = fanout
        self.dropout = 0.0

        self.aggregator = aggregator

        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.epochs_eval = epochs_eval
        self.optimizer = optimizer
        self.lr = lr

    def data_loader(self, device):
        g, _ = create_dgl_graph(self.dataset, self.path, self.multilabel)
        g.create_formats_()
        self.in_feats = g.ndata['feat'].shape[1]

        if multilabel:
            self.n_classes = len(g.ndata['label'][0])
        else:
            self.n_classes = len(np.unique(g.ndata['label']))

        train_nids = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        valid_nids = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
        test_nids = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]


        sampler = dgl.dataloading.NeighborSampler(self.fanout)
        self.train_dataloader = dgl.dataloading.DataLoader(
            g,
            train_nids,
            sampler,
            use_ddp=False,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )

        sampler2 = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        self.valid_dataloader = dgl.dataloading.DataLoader(
            g,
            valid_nids,
            sampler2,
            device=device,
            use_ddp=False,
            batch_size=512,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        self.test_dataloader = dgl.dataloading.DataLoader(
            g,
            test_nids,
            sampler2,
            device=device,
            use_ddp=False,
            batch_size=512,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

    def model_initial(self):
        if self.model == 'GraphSAGE':
            gnn_model = GraphSAGE(self.in_feats, self.n_hidden, self.n_layers, self.n_classes,
                                  self.dropout, self.aggregator, self.activation)
        if self.optimizer == 'sgd':
            opt = torch.optim.SGD(gnn_model.parameters(), lr=self.lr)
        elif self.optimizer == 'adam':
            opt = torch.optim.Adam(gnn_model.parameters(), lr=self.lr)
        else:
            print('No Support for \'{}\' Yet. Adam is used instead.'.format(self.optimizer))
            opt = torch.optim.Adam(gnn_model.parameters(), lr=self.lr)

        return gnn_model, opt

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

    def model_training(self, device):
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
                mfgs = [mfg.to(device) for mfg in mfgs]
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
            torch.save(model.state_dict(), self.output_path + self.model  + '-' + str(epoch) + '.pt')
            # Evaluate on only the first GPU.
            if epoch % self.epochs_eval == 0:
                print('Epoch: ', epoch)
                valid_accuracy, test_accuracy = self.evaluation(model)
                results.append([valid_accuracy, test_accuracy])

        best_result = max(results, key=lambda x: x[0])
        print('Best Accuracy: ', best_result)

    def run(self):
        print('=========='*5)
        print('Start GNN Training!')
        self.model_training('cuda')


if __name__ == "__main__":
    path = os.path.abspath(os.getcwd())
    dataset = 'cora'
    print(dataset)

    multilabel = False

    epochs = 100
    hidden = 128

    gnn_model = GNN(dataset = dataset, multilabel = multilabel,
                    path = path + '/datasets/',
                    output_path = path + '/output/' + 'centralized/' + dataset + '/',
                    model='GraphSAGE',
                    n_hidden= hidden,
                    n_layers = 2,
                    fanout = [25,10],
                    aggregator = 'mean',
                    activation = 'relu',
                    epochs = epochs, batch_size = 256, epochs_eval = 1,
                    optimizer = 'adam', lr = 0.01)
    gnn_model.run()

    print('====='*10)
