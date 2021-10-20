from collections import defaultdict
from collections.abc import Mapping

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import tqdm

class RelGATConvLayer(nn.Module):
    r"""Relational graph attention layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_heads : int
        Number of attention heads.
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_heads,
                 *,
                 activation=None,
                 dropout=0.0):
        super(RelGATConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_heads = num_heads
        self.activation = activation

        self.conv = dglnn.HeteroGraphConv({
                str(rel) : dglnn.GATConv(in_feat, out_feat, num_heads, allow_zero_in_degree=True)
                for rel in rel_names
            })

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs)

        def _apply(ntype, h):
            if self.activation:
                h = self.activation(h.view(-1, self.out_feat * self.num_heads))
            return self.dropout(h)
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}


class RGATEncoder(nn.Module):
    r"""
    RGAT Encoder Module. Extending GAT to heterogeneous graphs

    Parameters
    ----------
    etypes : list[str]
        list of edge types/relations in the graph
    in_size : int
        Size of input features
    hidden_size: int
        Size of output representations
    num_layers: int
        number of RGCN layers (Default: 1)
    num_heads: int
        number of bases for weight matrices (Default: 1)
    dropout: float
        dropout rate for feature and attention weights (Default: 0.)
    activation: Callable
        callable activation function/layer or None, optional. (Default: torch.nn.Functional.relu)
    """

    def __init__(self,
                 etypes,
                 in_size,
                 hidden_size,
                 num_layers=1,
                 num_heads=1,
                 dropout=0.,
                 activation=F.elu):
        super(RGATEncoder, self).__init__()

        self.in_dim = in_size
        self.h_dim = hidden_size
        self.rel_names = sorted(etypes)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()

        self.layers.append(RelGATConvLayer(self.in_dim, self.h_dim, self.rel_names,self.num_heads,
                                             activation=activation, dropout=self.dropout))

        for i in range(1, num_layers):
            self.layers.append(RelGATConvLayer(self.h_dim * self.num_heads, self.h_dim, self.rel_names, self.num_heads,
                                                 activation=activation, dropout=self.dropout))

    def forward(self, g, h):
        for layer, block in zip(self.layers, g):
            h = layer(block, h)
        return h



    def batch_inference(self, g, x, batch_size, device, num_workers):
        """Minibatch inference of final representation over all node types.
        ***NOTE***
        For node classification, the model is trained to predict on only one node type's
        label.  Therefore, only that type's final representation is meaningful.

        Parameters
        ----------
        g : dgl.DGLGraph
            the graph to predict on
        x: dict[str: th.Tensor]
            the input features/embeddings for all nodes
        batch_size : int
            Number of target nodes in a single batch
        device: str
            Device to use for computation
        num_workers: int
            number of dataloader workers
        """

        for l, layer in enumerate(self.layers):
            y = {k: th.zeros(g.number_of_nodes(k),self.h_dim*self.num_heads)
                for k in g.ntypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                {k: th.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)

                if not isinstance(input_nodes, Mapping):
                    input_nodes = {g.ntypes[0]: input_nodes}
                if not isinstance(output_nodes, Mapping):
                    output_nodes = {g.ntypes[0]: output_nodes}
                h = {k: x[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                h = layer(block, h)

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y