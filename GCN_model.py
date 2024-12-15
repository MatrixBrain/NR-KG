# GCN模型定义
import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Linear
import torch.nn as nn


# Define the NRKG model
class GCN(torch.nn.Module):
    def __init__(self, embeddings_num, edgetype_num, embeddings_dict, in_channels, hidden_channels, out_channels, num_layers, dropout, device, activation):
        super(GCN, self).__init__()

        self.device = device
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.embeddings = nn.Embedding(embeddings_num, hidden_channels)
        max_key = max(embeddings_dict.keys())
        embeddings_dict_tensor = torch.zeros((int(max_key+1), 1), device=device)
        for key, value in embeddings_dict.items():
            embeddings_dict_tensor[int(key)] = value
        self.embeddings_dict = embeddings_dict_tensor

        self.embeddings_lin1 = Linear(in_channels, int(hidden_channels/2))
        self.embeddings_lin2 = Linear(int(hidden_channels/2), hidden_channels)
        self.embeddings_lin3 = Linear(hidden_channels, hidden_channels)
        self.embeddings_lin_activation = F.mish

        self.embeddings_edge = nn.Embedding(edgetype_num, hidden_channels)

        self.attention = Linear(hidden_channels, 1, bias=False)

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin1 = Linear(hidden_channels, int(hidden_channels/2), bias=True)
        self.lin2 = Linear(int(hidden_channels/2), out_channels, bias=True)
        self.dropout = dropout

        if activation == 'ReLU':
            self.activation = F.relu
        elif activation == 'LeakyReLU':
            self.activation = F.leaky_relu
        elif activation == 'PReLU':
            self.preluweight = torch.nn.Parameter(torch.Tensor(hidden_channels))
            self.preluweights = torch.nn.Parameter(torch.Tensor(hidden_channels))
            self.preluweightlin1 = torch.nn.Parameter(torch.Tensor(int(hidden_channels/2)))
            self.activation = F.prelu
        elif activation == 'Tanh':
            self.activation = F.tanh
        elif activation == 'ELU':
            self.activation = F.elu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the parameters"""
        bound = 6. / math.sqrt(self.hidden_channels)
        nn.init.uniform_(self.embeddings.weight, -bound, bound)
        nn.init.uniform_(self.embeddings_edge.weight, -bound, bound)
        self.embeddings_edge.weight.data = F.normalize(self.embeddings_edge.weight.data, p=2, dim=-1)
        self.embeddings_lin1.reset_parameters()
        self.embeddings_lin2.reset_parameters()
        self.embeddings_lin3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, graph, batch):
        x = graph.x
        x_id = graph.x_id
        x_attr = graph.x_attr
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr.long()
        weight = graph.edge_weight
        x = self.embeddings_lin1(x)
        x = self.embeddings_lin_activation(x)
        x = self.embeddings_lin2(x)
        x = self.embeddings_lin_activation(x)
        x = self.embeddings_lin3(x)

        hea_id = torch.where(x_attr == 1)[0]
        not_hea_id = torch.where(x_attr != 1)[0]
        x[not_hea_id] = self.embeddings(self.embeddings_dict[x_id[not_hea_id].long().reshape(-1)].long()).reshape(-1, self.hidden_channels)

        x = F.normalize(x, p=2, dim=-1) 
        head = x[edge_index[0]]
        relation = self.embeddings_edge(edge_attr).reshape(-1, self.hidden_channels)
        tail_estimate = head + relation

        edge_index_bool = (edge_index[1].reshape(-1, 1).expand(-1, len(hea_id))) == (hea_id.reshape(1, -1).expand(edge_index.shape[1], -1))
        edge_index_bool = edge_index_bool.reshape(edge_index_bool.shape[0], 1, edge_index_bool.shape[1]).expand(edge_index_bool.shape[0], tail_estimate.shape[1], edge_index_bool.shape[1])

        tail_estimate = tail_estimate.reshape(tail_estimate.shape[0], tail_estimate.shape[1], 1).expand(tail_estimate.shape[0], tail_estimate.shape[1], hea_id.shape[0])
        tail_estimate = torch.sum(tail_estimate*edge_index_bool, dim=0) / torch.sum(edge_index_bool, dim=0)[0].reshape(1, -1)
        tail_estimate = tail_estimate.t()
        score_semantic = torch.norm(x[hea_id] - tail_estimate, p=2, dim=-1).reshape(-1, 1)

        head = x[edge_index[0]]
        tail = x[edge_index[1]]
        relation = self.embeddings_edge(edge_attr).reshape(-1, self.hidden_channels)
        pos_score = torch.norm(head + relation - tail, p=2, dim=-1) 

        neg_head = x[torch.randint(x.shape[0], (edge_index.shape[1],))]
        neg_score = torch.norm(neg_head + relation - tail, p=2, dim=-1)

        x1 = x.clone()
        x = self.conv1(x, edge_index, weight)
        if self.activation == F.prelu:
            x = self.activation(x, self.preluweight)
        else:
            x = self.activation(x)
        for conv in self.convs:
            x = conv(x, edge_index, weight)
            if self.activation == F.prelu:
                x = self.activation(x, self.preluweights)
            else:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + x1

        x = self.lin1(x)
        if self.activation == F.prelu:
            x = self.activation(x, self.preluweightlin1)
        else:
            x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x, (pos_score, neg_score, score_semantic)

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add') 
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, fill_value=2, edge_weight=None):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

        x = self.lin(x)
        x = F.relu(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)

    def message(self, x_j, norm, edge_weight):
        return norm.view(-1, 1) * x_j if edge_weight is None else norm.view(-1, 1) * edge_weight.view(-1, 1) * x_j
    
if __name__ == '__main__':
    pass