import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, num_classes, num_layers, aggr):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_dimension, hidden_dimension, aggr=aggr))
        for i in range(self.num_layers-2):
            self.layers.append(SAGEConv(hidden_dimension, hidden_dimension, aggr=aggr))
        self.layers.append(SAGEConv(hidden_dimension, num_classes, aggr=aggr))

    def forward(self, h, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            h_target = h[:size[1]]
            h = self.layers[i]((h, h_target), edge_index)
            if i != self.num_layers - 1:
                h = F.relu(h)
        return h

    def inference(self, h_all):
        for i in range(self.num_layers):
            hs = []
            for _, n_id, adj in test_loader:
                edge_index, _, size = adj.to(device)
                h = h_all[n_id].to(device)
                h_target = h[:size[1]]
                h = self.layers[i]((h, h_target), edge_index)
                if i != self.num_layers - 1:
                    h = F.relu(h)
                hs.append(h)

            h_all = torch.cat(hs, dim=0)

        return h_all

def Train(g, model, train_loader, lr, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_test_acc = 0

    model.train()
    for epoch in range(1, n_epochs+1):

        for batch_size, n_id, adjs in train_loader:
            adjs = [adj.to(device) for adj in adjs]
            ## Forward
            out = model(g.x[n_id], adjs)
            ## Compute loss
            loss = F.cross_entropy(out, g.y[n_id[:batch_size]])
            ## Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ## Compute prediction
        model.eval()
        with torch.no_grad():
            preds = model.inference(g.x).argmax(dim=1)

        ## Compute accuracy on training/validation/test
        train_acc = (preds[g.train_mask] == g.y[g.train_mask]).float().mean()
        val_acc = (preds[g.val_mask] == g.y[g.val_mask]).float().mean()
        test_acc = (preds[g.test_mask] == g.y[g.test_mask]).float().mean()

        ## Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 1 == 0:
            print('Epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc))

    print('best_val_acc: {:.3f}, best_test_acc: {:.3f}'.format(best_val_acc, best_test_acc))

if __name__ == '__main__':

    dataset = torch_geometric.datasets.Planetoid(root='/tmp/Cora', name='Cora', split='public')

    g = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = g.to(device)

    lr = 0.01
    hidden_d = 100
    n_epochs = 100
    batch_size = 256
    n_layers = 2
    fan_out = [25,10]

    print(f"lr={lr}, hidden_d={hidden_d}, n_epochs={n_epochs}, batch_size={batch_size}, n_layers={n_layers}, fan_out={fan_out}")

    train_loader = torch_geometric.loader.NeighborSampler(g.edge_index, node_idx=g.train_mask,
                                sizes=fan_out, batch_size=256, shuffle=True, num_workers=0)

    test_loader = torch_geometric.loader.NeighborSampler(g.edge_index, node_idx=None, sizes=[-1],
                                    batch_size=256, shuffle=False, num_workers=0)

    model = GraphSAGE(dataset.num_node_features, hidden_d, dataset.num_classes, n_layers, 'mean').to(device)

    Train(g, model, train_loader, lr=lr, n_epochs=n_epochs)

    print('====='*10)
