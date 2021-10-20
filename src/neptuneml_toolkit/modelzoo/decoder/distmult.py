import torch as th
import torch.nn as nn

class DistmultDecoder(nn.Module):
    r"""
    Implementation of Distmult Decoder.

    Paper link: https://arxiv.org/abs/1412.6575

    Suitable when predicting a logit for each edge where
    edge is represented by a pair of arrays of node features
    and optionally an edge type id

    .. math::

        f(u,r,v) = \textbf{u}^{T} \textbf{W}_r \textbf{v} = \sum_{i=1}^{d}\textbf{u}_i \cdot
        diag(\textbf{W}_r)_i \cdot \textbf{v}_i

    Parameters
    ----------
    in_units : int
        Size of input node features
    num_rels: int
        Number of distinct edge types (Default: 1)
    """
    def __init__(self, in_units, num_rels=1):
        super(DistmultDecoder, self).__init__()
        self.w_relation = nn.Parameter(th.Tensor(num_rels, in_units))
        self.num_rels = num_rels
        nn.init.xavier_uniform_(self.w_relation)

    def forward(self, u, v, etype_ids=None):
        """Forward function.
        Compute logits for each pair ``(ufeat[i], vfeat[i])``.
        Parameters
        ----------
        u : th.Tensor
            Source nodes embeddings. Shape: (B, D)
        v : th.Tensor
            Destination nodes embeddings. Shape: (B, D)
        etype_ids: th.Tensor
            Edge type ids for each edge represented by u[i], v[i]
        Returns
        -------
        th.Tensor
            Representation for edge between u and v. Shape: (B, out)
        """
        assert u.shape == v.shape, "Shape mismatch between u and v"
        if etype_ids is None:
            assert self.num_rels == 1, "You must pass in etype_ids if num_rels is more than 1"
        r_emb = self.w_relation[etype_ids] if etype_ids is not None else self.w_relation[0]
        score = th.sum(u * r_emb * v, dim=-1)
        return score

class GraphDistmultDecoder(nn.Module):
    r"""
    Implementation of Distmult Decoder using DGL message passing API.

    Paper link: https://arxiv.org/abs/1412.6575

    Suitable when predicting a logit for each edge using the graph structure

    .. math::

        f(u,r,v) = \textbf{u}^{T} \textbf{W}_r \textbf{v} = \sum_{i=1}^{d}\textbf{u}_i \cdot
        diag(\textbf{W}_r)_i \cdot \textbf{v}_i

    Parameters
    ----------
    in_units : int
        Size of input node features
    num_rels: int
        Number of distinct edge types (Default: 1)
    """
    def __init__(self, in_units, num_rels=1):
        super(GraphDistmultDecoder, self).__init__()
        self.w_relation = nn.Parameter(th.Tensor(num_rels, in_units))
        self.num_rels = num_rels
        nn.init.xavier_uniform_(self.w_relation)

    def forward(self, graph, h):
        """Forward function.
        Compute output for each edge ``(u, v)`` in the graph using `h_u` and h_v`.
        Parameters
        ----------
        graph : dgl.Graph
            The graph structure with the same number of nodes as h
        h : th.Tensor or dict[ntype: th.Tensor]
            Nodes embeddings. Shape: (N, D)
        Returns
        -------
        th.Tensor
            Representation for edge between u and v. Shape: (B, out)
        """
        with graph.local_scope():
            graph.ndata['h'] = h
            for etype in graph.canonical_etypes:
                if graph.num_edges(etype):
                    graph.apply_edges(self.distmult_udf(graph.get_etype_id(etype)), etype=etype)
            return graph.edata['out']

    def distmult_udf(self, etype_id):
        def udf(edges):
            return {'out': th.sum(edges.src['h'] * self.w_relation[etype_id] * edges.dst['h'], dim=-1)}
        return udf