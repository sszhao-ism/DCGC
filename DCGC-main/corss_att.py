
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
#from utils import glorot_init
from utils import *

class AGCN(nn.Module):
    def __init__(self, num_nodes): 
        super(AGCN, self).__init__()
        self.n = num_nodes
        self.w1 = Parameter(torch.FloatTensor(self.n,self.n))
        self.w1.data = torch.eye(self.n)
        self.w2 = Parameter(torch.FloatTensor(self.n,self.n))
        self.w2.data = torch.eye(self.n)
    
    def forward(self, X, A):
        H = torch.mm(torch.mm(A, self.w1), A.T)   
        H = torch.mm(torch.mm(H, self.w2), X)
        embed = torch.mm(H, H.T)  
        embed = F.normalize(embed, dim=1) 

        return embed


class GCN(nn.Module):
    def __init__(self, input_dim, activation = F.relu, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, input_dim)
        self.activation = activation

    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

class AGCN_Net(nn.Module):
    def __init__(self, N,  args):
        super().__init__()
        
        self.n_layers = args.n_layers

        self.AttenGCN = AGCN(N)
        
        self.extractor = nn.ModuleList()
        self.extractor.append(nn.Linear(N, args.hid_dim))
        for i in range(self.n_layers - 1):
            self.extractor.append(nn.Linear(args.hid_dim, args.hid_dim))
        self.dropout = nn.Dropout(p=args.dropout)

        self.init_weights()
        
        self.params_exp = list(self.AttenGCN.parameters())  \
                            + list(self.extractor.parameters()) 

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)

    def forward(self, knn, adj):
        h = self.AttenGCN(knn,adj)
        for i, layer in enumerate(self.extractor):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
        return h


def add_self_loops(adj_matrix):
    if adj_matrix is None:
        raise ValueError("adj_matrix cannot be None")
    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    num_nodes = adj_matrix.size(0)
    adj_matrix = adj_matrix + torch.eye(num_nodes, dtype=adj_matrix.dtype, device=adj_matrix.device)
    return adj_matrix

