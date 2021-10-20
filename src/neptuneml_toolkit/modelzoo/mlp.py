import torch.nn as nn

class MLP(nn.Module):
    r"""
   Implementation of MLP module.

   Parameters
   ----------
   in_units : int
       Size of input features
   out_units: int
       Size of output units
   num_layers : int
       number of layers in mlp (Default: 1)
   dropout_rate: float
       dropout rate, dropout is applied before every layer (Default: 0.)
   hidden_units: int or list[int]
       size of hidden units or list of hidden unit sizes for each hidden layer (Default: None).
       By default set to the in_units
   bias: bool
        whether to use bias in each layer (Default: true)
   """

    def __init__(self,
                 in_units,
                 out_units,
                 num_layers=1,
                 dropout_rate=0,
                 hidden_units=None,
                 bias=True):

        super(MLP, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList()
        units = [in_units]
        if hidden_units:
            assert isinstance(hidden_units, list) or isinstance(hidden_units, int), "Unsupported type for hidden_units"
        else:
            hidden_units = in_units

        if isinstance(hidden_units, int):
            hidden_units = [hidden_units] * (num_layers-1)
        else:
            assert len(hidden_units) == num_layers - 1, "Mismatch between hidden units per layer and number of layers"

        units += hidden_units
        units.append(out_units)

        for in_unit, out_unit in zip(units[:-1], units[1:]):
            self.layers.append(nn.Linear(in_unit, out_unit, bias=bias))


    def forward(self, x):
        """Forward function.
        Apply mlp on input x``.
        Parameters
        ----------
        x : th.Tensor
            input. Shape: (B, D)
        Returns
        -------
        th.Tensor
            Representation for edge between u and v. Shape: (B, out)
        """
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)

        return x

