import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # 导入 GATConv 层


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, output_dim, dropout, heads=8):
        """
        GAT model, including multiple GATConv layers
        NFT: Dimension of input features
        Nhid: Number of hidden layer nodes (i.e. the output feature dimension of each GATConv layer)
        Output Dimension: Dimension of Output Node Features
        Dropout: dropout ratio
        Heads: the number of heads for multi head attention
        """
        super(GAT, self).__init__()

        # 定义两层 GATConv
        self.gat1 = GATConv(nfeat, nhid, heads=heads, dropout=dropout)
        self.gat2 = GATConv(nhid * heads, output_dim, heads=1, dropout=dropout)

    def forward(self, x, adj):
        x = F.elu(self.gat1(x, adj))
        x = self.gat2(x, adj)
        return x


