import torch as th
import torch.nn as nn
import dgl.function as fn

class DotProductDecoder(nn.Module):
    r"""
    Implementation of DotProduct Decoder.

    Suitable when predicting a logit for each edge where
    edge is represented by a pair of arrays of node features

    .. math::

        f(u,v) = \textbf{u}^{T}\textbf{v} = \sum_{i=1}^{d}\textbf{u}_i \cdot \textbf{v}_i

    """
    def __init__(self):
        super(DotProductDecoder, self).__init__()

    def forward(self, u, v):
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
        score = th.einsum('bi,bi->b', u, v)
        return score

class GraphDotProductDecoder(nn.Module):
    r"""
    Implementation of DotProduct Decoder using DGL Message Passing API.

    Suitable when predicting a logit for each edge where
    edge is represented by a pair of arrays of node features

    .. math::

        f(u,v) = \textbf{u}^{T}\textbf{v} = \sum_{i=1}^{d}\textbf{u}_i \cdot \textbf{v}_i

    """

    def __init__(self):
        super(GraphDotProductDecoder, self).__init__()

    def forward(self, graph, h):
        """Forward function.
        Compute output for each edge ``(u, v)`` in the graph using `h_u` and h_v`.
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
        with graph.local_scope():
            graph.ndata['h'] = h
            for etype in graph.canonical_etypes:
                graph.apply_edges(fn.u_dot_v('h', 'h', 'out'), etype=etype)
            return graph.edata['out']