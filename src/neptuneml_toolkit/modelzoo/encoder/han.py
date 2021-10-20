"""
Adapted from https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model_hetero.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    r"""
    Semantic Attention for HAN model.

    Parameters
    ----------
    in_size : int
        Size of input features
    hidden_size : int
        Size of hidden representation
    """
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        """

        Parameters
        ----------
        z : th.Tensor
            input. Shape (N, M, D*K)
        Returns
        -------
        th.Tensor
            Aggregate representation for all metapaths. Shape: (N, D * K)
        """
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    r"""
    HAN layer.

    Parameters
    ----------
    num_metapaths : int
        Number of metapaths
    in_size : int
        Size of input features
    out_size: int
        Size of output representations
    layer_num_heads: int
        number of attention heads (Default: 4)
    dropout: float
        dropout rate for feature and attention weights (Default: 0.)
    activation: Callable
        callable activation function/layer or None, optional. (Default: torch.nn.Functional.elu)
    """
    def __init__(self, num_metapaths, in_size, out_size, layer_num_heads=4, dropout=0., activation=F.elu):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=activation,
                                           allow_zero_in_degree=True))
        self.num_metapaths = num_metapaths

    def forward(self, graph, h):
        """Forward function.
        Compute output of han layer for input node features h
        Parameters
        ----------
        graph : dgl.Graph or dgl.Block
            The graph structure with the same number of nodes as h
        h : th.Tensor or dict[ntype: th.Tensor]
            Nodes embeddings. Shape: (N, D)
        Returns
        -------
        th.Tensor
            Representation for edge between u and v. Shape: (B, out)
        """
        metapath_embeddings = []
        for i, g in enumerate(graph):
            metapath_embeddings.append(self.gat_layers[i](g, h[i]).flatten(1))
        return  metapath_embeddings

class HANEncoder(nn.Module):
    r"""
    HAN Encoder Module.

    Paper link: https://github.com/Jhy1993/Representation-Learning-on-Heterogeneous-Graph

    Current implementation supports only node classification tasks

    Parameters
    ----------
    num_metapaths : list[str] or list[tuple[str]]
        Metapath in the form of a list of edge types
    in_size : int
        Size of input features
    hidden_size: int
        Size of output representations
    num_layers: int
        number of HAN layers
    num_heads: int or list[int]
        number of attention heads for each layer (Default: 4)
    dropout: float
        dropout rate for feature and attention weights (Default: 0.)
    activation: Callable
        callable activation function/layer or None, optional. (Default: torch.nn.Functional.elu)
    """
    def __init__(self, num_metapaths, in_size, hidden_size, num_layers, num_heads=4, dropout=0., activation=F.elu):
        super(HANEncoder, self).__init__()

        self.layers = nn.ModuleList()
        if isinstance(num_heads, int):
            num_heads = [num_heads] * (num_layers)
        else:
            assert len(num_heads) == num_layers, "Mismatch between attention heads for each layer and number of layers"
        self.layers.append(HANLayer(num_metapaths, in_size, hidden_size, num_heads[0], dropout, activation))
        for l in range(1, num_layers):
            self.layers.append(HANLayer(num_metapaths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout, activation))
        self.semantic_attention = SemanticAttention(in_size=hidden_size * num_heads[l], hidden_size=hidden_size)
        self.project = nn.Linear(hidden_size * num_heads[-1], hidden_size)

    def forward(self, g, h):
        r"""
        Compute the node representations on initial node features with graph g

        Parameters
        ----------
        g : list[dgl.Graph/dgl.Block] or list[list[dgl.Graph/dgl.Block]]
            The list of graphs for each metapath. For more than one layers
            each entry in the list should be list of message flow graphs for each metapath
        h: list[th.Tensor]
            The list of initial node features for the input nodes for each metapath

        """
        for l, gnn in enumerate(self.layers):
            mfg = [block[l] for block in g] if type(g[0]) == list else g
            h = gnn(mfg, h)
        semantic_embeddings = self.semantic_attention(torch.stack(h, dim=1))  # (N, D * K)
        return self.project(semantic_embeddings)