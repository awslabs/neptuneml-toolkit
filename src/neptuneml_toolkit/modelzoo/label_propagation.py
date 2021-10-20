import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

from collections.abc import Sequence

class LabelPropagation(nn.Module):
    r"""
    Implementation of Label Propagation module
    Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_
    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},
    where unlabeled data is inferred by labeled data via propagation.
    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
        adj_norm: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
    """

    def __init__(self, num_layers, alpha, adj_norm='DAD'):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
        self.adj = adj_norm

    @torch.no_grad()
    def forward(self, g, labels, mask=None, clamp=True):
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)

            y = labels
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]

            weighted_label = (1 - self.alpha) * y
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5 if self.adj == 'DAD' else -1).to(labels.device).unsqueeze(1)

            for _ in range(self.num_layers):
                # Assume the graphs to be undirected
                if self.adj in ['DAD', 'AD']:
                    y = norm * y

                g.ndata['h'] = y
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = self.alpha * g.ndata.pop('h')

                if self.adj in ['DAD', 'DA']:
                    y = y * norm

                y = weighted_label + y
                if clamp:
                    y = y.clamp_(0., 1.)

        return y
