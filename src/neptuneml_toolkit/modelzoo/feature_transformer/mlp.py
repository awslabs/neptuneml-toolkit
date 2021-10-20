import torch as th
import torch.nn as nn
from collections.abc import Mapping
from ..mlp import MLP

class MLPFeatureTransformer(nn.Module):
    r"""
   Implementation of MLP feature transformer.

   Transforms a feature dictionary per node type to a feature Tensor per node type by concatenating
   and projecting with an MLP. Optionally, process each feature per node type before concatenating.

   Parameters
   ----------
   in_units : dict[ntype: dict[feat_name: int]]
       Size of input features
   out_units: int
       Size of output units
   num_layers : int or dict[ntype: int]
       number of layers in mlp (Default: 1)
   per_feat_name: bool,
       embed per feature name (Default: False)
   out_unit_per_feat: int
       used if per feature name is set to True (Default: 16)
   dropout_rate: float
       dropout rate, dropout is applied before every layer (Default: 0.)
   hidden_units: int or dict[ntype: int] or dict[ntype: list[int]]
       size of hidden units or list of hidden unit sizes for each hidden layer (Default: None).
       By default set to the in_units
   bias: bool
        whether to use bias in each layer (Default: true)
   """

    def __init__(self,
                 in_units,
                 out_units,
                 num_layers=1,
                 per_feat_name=False,
                 out_unit_per_feat=16,
                 dropout_rate=0,
                 hidden_units=None,
                 bias=True):

        super(MLPFeatureTransformer, self).__init__()

        self.in_units = in_units
        self.out_units = out_units
        self.per_feat_name = per_feat_name
        self.out_units_per_feat = out_unit_per_feat
        self.per_feat_modules = nn.ModuleDict()
        self.ntype_feat_modules = nn.ModuleDict()

        for ntype, feature_dims in in_units.items():
            in_size = 0
            for feature_name, feature_dim in feature_dims.items():
                if self.per_feat_name:
                    self.per_feat_modules[str((ntype, feature_name))] = MLP(feature_dim, out_unit_per_feat,
                                                                            num_layers=1,
                                                                            dropout_rate=dropout_rate, bias=bias)
                    in_size += out_unit_per_feat
                else:
                    in_size += feature_dim

            self.ntype_feat_modules[ntype] = MLP(in_size, out_units, num_layers=num_layers, dropout_rate=dropout_rate,
                                                 bias=bias)


    def forward(self, x):
        h = {}
        for ntype in x:
            ntype_h = []
            for feat_name, feat in x[ntype].items():
                if self.per_feat_name:
                    ntype_h.append(self.per_feat_modules[str((ntype, feat_name))](feat))
                else:
                    ntype_h.append(feat)
            h[ntype] = self.ntype_feat_modules[ntype](th.cat(ntype_h, dim=1))

        return h

