import torch as th
from torch.utils.data import DataLoader

import dgl
from dgl import utils

from collections.abc import Mapping, Sequence

class MultiSubgraphNodeDataLoader:
    r"""
    Pytorch Dataloader to be used with a Graph Sampler that returns multiple
    distinct subgraph views

    Parameters
    ----------
    g : DGLGraph
        heterogeneous dgl graph
    nids: th.Tensor or dict[str: th.Tensor]
        the dataset node ids to sample from
    sampler: int
        the sampler to generate multiple subgraph views for each batch
    """

    def __init__(self, g, nids, sampler, device=None, **kwargs):
        if device is None:
            # default to the same device the graph is on
            device = th.device(g.device)
        self.collator = MultiSubgraphNodeCollator(g, nids, sampler, device)
        self.dataloader = DataLoader(
            self.collator.dataset,
            collate_fn=self.collator.collate,
            **kwargs)

    def __iter__(self):
        """Return the iterator of the data loader."""
        return iter(self.dataloader)

    def __len__(self):
        """Return the number of batches of the data loader."""
        return len(self.dataloader)

class MultiSubgraphNodeCollator:
    r"""
    NodeCollator for multi-subgraph graph sampler

    At each minibatch multiple subgraphs are returned by the sampler
    and each subgraph is a different view of the original graph
    around the current minibatch seeds

    Parameters
    ----------
    g : DGLGraph
        heterogeneous dgl graph
    nids: th.Tensor or dict[str: th.Tensor]
        the dataset node ids to sample from
    sampler: int
        the sampling method to use to generate multiple subgraph views
    """
    def __init__(self, g, nids, sampler, device=None):
        self.g = g
        self.sampler = sampler
        self.device = device
        if isinstance(nids, Mapping):
            self._dataset = utils.FlattenedDict(nids)
        else:
            self._dataset = utils.prepare_tensor(g, nids, 'nids')

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        r"""
        Sample subgraphs and collate for mini-batch training

        Parameters
        ----------
        items : list[int] or list[tuple[str, int]]
            starting seed nodes

        Returns
        -------
         tuple(list[th.Tensor] or list[dict[str: th.Tensor]], th.Tensor or dict[str: th.Tensor], list[DGLGraph])
            i.e (input_node_list, output_nodes, subgraph_list)
            a tuple with list of input node ids for each subgraph,
            the target node ids and
            a list of MSGs for each subgraph
        """
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            seed_nodes = {ntype: th.stack(seeds).reshape(-1) for ntype, seeds in utils.group_as_dict(items).items()}
        else:
            seed_nodes = utils.prepare_tensor(self.g, items, 'items')
        blocks = self.sampler.sample_blocks(seed_nodes)
        output_nodes = []
        input_nodes = []
        for block in blocks:
            if isinstance(block, list): # check if this is a multi-layer MFG sequence
                input_nodes.append(block[0].srcdata[dgl.NID])
                output_nodes.append(block[-1].dstdata[dgl.NID])
            else:
                input_nodes.append(block.srcdata[dgl.NID])
                output_nodes.append(block.dstdata[dgl.NID])
        if self.device is not None:
            input_nodes = [{k: v.to(self.device) for k,v in input_node.items()} if isinstance(input_node, Mapping)
                           else input_node.to(self.device) for input_node in input_nodes]
            output_nodes = [{k: v.to(self.device) for k,v in output_node.items()} if isinstance(output_node, Mapping)
                           else output_node.to(self.device) for output_node in output_nodes]
            blocks = [[b.to(self.device) for b in block] if isinstance(block, Sequence)
                      else block.to(self.device) for block in blocks]
        return input_nodes, output_nodes, blocks