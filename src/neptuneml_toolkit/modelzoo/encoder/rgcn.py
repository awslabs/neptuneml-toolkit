"""RGCN implementation adapted from https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/model.py
"""

from collections import defaultdict
from collections.abc import Mapping

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import tqdm

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
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
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                str(rel) : dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
                for rel in rel_names
            })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

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
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i] : {'weight' : w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}

class RGCNEncoder(nn.Module):
    r"""
    RGCN Encoder Module.

    Paper link: https://github.com/Jhy1993/Representation-Learning-on-Heterogeneous-Graph

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
    num_bases: int
        number of bases for weight matrices (Default: -1 [use all etypes])
    use_self_loop: bool
        whether to use self loops (Default: True)
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
                 num_bases=-1,
                 use_self_loop=True,
                 dropout=0.,
                 activation=F.relu):
        super(RGCNEncoder, self).__init__()

        self.in_dim = in_size
        self.h_dim = hidden_size
        self.rel_names = sorted(etypes)
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.layers = nn.ModuleList()

        self.layers.append(RelGraphConvLayer(self.in_dim, self.h_dim, self.rel_names,self.num_bases,
                                             activation=activation, self_loop=self.use_self_loop,
                                             dropout=self.dropout))

        for i in range(1, num_layers):
            self.layers.append(RelGraphConvLayer(self.h_dim, self.h_dim, self.rel_names, self.num_bases,
                                                 activation=activation, self_loop=self.use_self_loop,
                                                 dropout=self.dropout))

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
            y = {k: th.zeros(g.number_of_nodes(k),self.h_dim)
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