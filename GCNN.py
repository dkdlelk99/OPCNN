import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.nn import global_mean_pool


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_bn=True):
        super(GCN, self).__init__()
        self.L = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(self.L - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(GCNConv(hidden_channels, out_channels))
        self.use_bn = use_bn
        self.lin = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, batch):
        xs = []  # layer0, layer1, layer2, mean_pool
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x) # intermediate results
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x) # intermediate results
        
        x = global_mean_pool(x, batch)
        xs.append(x) # intermediate results
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x

    def generate_latent_vector(self, x, edge_index, batch):
        xs = []  # layer0, layer1, layer2, mean_pool
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x) # intermediate results
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x) # intermediate results
        
        x = global_mean_pool(x, batch)
        xs.append(x) # intermediate results
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.lin(x)
        
        return x