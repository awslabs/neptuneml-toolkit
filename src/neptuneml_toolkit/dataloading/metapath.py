import numpy as np
from collections.abc import Mapping, Sequence

import dgl
from dgl import backend as F, convert, transform
from dgl.sampling.randomwalks import random_walk
from dgl.sampling.neighbor import select_topk
from dgl.base import EID
from dgl import utils

class RandomWalkNeighborSampler(object):
    """PinSage-like neighbor sampler extended to any heterogeneous graphs.
    Given a heterogeneous graph and a list of nodes, this callable will generate a subgraph
    graph where the neighbors of each given node are the most commonly visited nodes of at the end of a metapath
    by performing multiple random walks starting from that given node.  Each random walk consists
    of , with a probability of termination after each traversal.

    This is a generalization of PinSAGE sampler which only works on bidirectional bipartite
    graphs.
    Parameters
    ----------
    G : DGLGraph
        The graph.  It must be on CPU.
    num_traversals : int
        The maximum number of metapath-based traversals for a single random walk.
        Usually considered a hyperparameter.
    termination_prob : float
        Termination probability after each metapath-based traversal.
        Usually considered a hyperparameter.
    num_random_walks : int
        Number of random walks to try for each given node.
        Usually considered a hyperparameter.
    num_neighbors : int
        Number of neighbors (or most commonly visited nodes) to select for each given node.
    metapath : list[str] or list[tuple[str, str, str]], optional
        The metapath.
        If not given, DGL assumes that the graph is homogeneous and the metapath consists
        of one step over the single edge type.
    weight_column : str, default "weights"
        The name of the edge feature to be stored on the returned graph with the number of
        visits.
    Examples
    --------
    See examples in :any:`PinSAGESampler`.
    """
    def __init__(self, G, num_traversals, termination_prob,
                 num_random_walks, num_neighbors, metapath=None, weight_column='weights'):
        assert G.device == F.cpu(), "Graph must be on CPU."
        self.G = G
        self.weight_column = weight_column
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.num_traversals = num_traversals

        if metapath is None:
            if len(G.ntypes) > 1 or len(G.etypes) > 1:
                raise ValueError('Metapath must be specified if the graph is homogeneous.')
            metapath = [G.canonical_etypes[0]]
        self.start_ntype = G.to_canonical_etype(metapath[0])[0]
        self.end_ntype = G.to_canonical_etype(metapath[-1])[-1]

        self.metapath_hops = len(metapath)
        self.metapath = metapath
        self.full_metapath = metapath * num_traversals
        restart_prob = np.zeros(self.metapath_hops * num_traversals)
        restart_prob[self.metapath_hops::self.metapath_hops] = termination_prob
        self.restart_prob = F.zerocopy_from_numpy(restart_prob)

    # pylint: disable=no-member
    def __call__(self, seed_nodes):
        """
        Parameters
        ----------
        seed_nodes : Tensor
            A tensor of given node IDs of node type ``ntype`` to generate neighbors from.  The
            node type ``ntype`` is the beginning and ending node type of the given metapath.
            It must be on CPU and have the same dtype as the ID type of the graph.
        Returns
        -------
        g : DGLGraph
            A homogeneous graph constructed by selecting neighbors for each given node according
            to the algorithm above.  The returned graph is on CPU.
        """
        seed_nodes = utils.prepare_tensor(self.G, seed_nodes, 'seed_nodes')

        seed_nodes = F.repeat(seed_nodes, self.num_random_walks, 0)
        paths, _ = random_walk(
            self.G, seed_nodes, metapath=self.full_metapath, restart_prob=self.restart_prob)
        src = F.reshape(paths[:, self.metapath_hops::self.metapath_hops], (-1,))
        dst = F.repeat(paths[:, 0], self.num_traversals, 0)

        src_mask = (src != -1)
        src = F.boolean_mask(src, src_mask)
        dst = F.boolean_mask(dst, src_mask)

        # count the number of visits and pick the K-most frequent neighbors for each node
        if self.start_ntype == self.end_ntype:
            self.ntype = self.start_ntype
            neighbor_graph = convert.heterograph(
                {(self.ntype, '_E', self.ntype): (src, dst)},
                {self.ntype: self.G.number_of_nodes(self.ntype)}
            )
        else:
            neighbor_graph = convert.heterograph(
                {(self.end_ntype, '_E', self.start_ntype): (src, dst)},
                {self.start_ntype: self.G.number_of_nodes(self.start_ntype),
                 self.end_ntype: self.G.number_of_nodes(self.end_ntype)}
            )

        neighbor_graph = transform.to_simple(neighbor_graph, return_counts=self.weight_column)
        counts = neighbor_graph.edata[self.weight_column]
        neighbor_graph = select_topk(neighbor_graph, self.num_neighbors, self.weight_column)
        selected_counts = F.gather_row(counts, neighbor_graph.edata[EID])
        neighbor_graph.edata[self.weight_column] = selected_counts

        return neighbor_graph

class MetapathListSampler(object):
    r"""
    Sample subgraphs for a list of metapaths by perform random walks for each metapath.

    Parameters
    ----------
    g : DGLGraph
        heterogeneous dgl graph
    metapath_list : list[list[str or tuple of str]]
        list of metapaths. each metapath is specified as a list of edge_types or canonical_edge_types.
    num_neighbors: int or list, optional
        Number of neighbors (or most commonly visited nodes) to select for each given node. (Default: 1)
    num_random_walks : int, optional
        Number of random walks to try for each given node. (Default: None)
        This is set ot num_neighbours by default. If num_random_walks > num_neighbors and the
        random walks find more than num_neighbours neighbours for a particular node the fanout
        for that node is still capped at num_neighbors
    dropout_rate : float, optional
        Dropout rate (Default: 0.0)
    add_self_loop: bool
    """
    def __init__(self, g, metapath_list, num_neighbors=1, num_random_walks=None):
        metapath_list = [[g.to_canonical_etype(etype) for etype in metapath] for metapath in metapath_list]
        if isinstance(num_neighbors, Sequence):
            for metapath in metapath_list:
                assert metapath[0][0] == metapath[-1][-1], \
                    "For multilayer sampling all metapaths must start and end at the same node type"
        if not isinstance(num_neighbors, Sequence):
            assert type(num_neighbors) == int
            num_neighbors = [num_neighbors]
        if not num_random_walks:
            num_random_walks = num_neighbors
        else:
            assert type(num_random_walks) == int
            num_random_walks = [num_random_walks]*len(num_neighbors)
        self.sampler_list = []
        self.metapaths_src_ntype = []
        for metapath in metapath_list:
            metapath_layer_sampler_list = []
            for neighbour_num, rw_num in zip(num_neighbors, num_random_walks):
                metapath_layer_sampler_list.append(RandomWalkNeighborSampler(G=g,
                                                                           num_traversals=1,
                                                                           termination_prob=0,
                                                                           num_random_walks=rw_num,
                                                                           num_neighbors=neighbour_num,
                                                                           metapath=metapath))
            self.sampler_list.append(metapath_layer_sampler_list)
            self.metapaths_src_ntype.append(g.to_canonical_etype(metapath[0])[0])
        self.num_target_ntypes = len(set(self.metapaths_src_ntype))

    def sample_blocks(self, seeds):
        r"""
        Perform one round of sampling for seed nodes
        Parameters
        ----------
        seeds : th.Tensor or dict[ntype: th.Tensor]
            seed nodes of type 'ntype' where ``ntype`` is the ending node type of the given metapath

        Returns
        -------
        (th.Tensor, list[DGLGraph])
            the input seeds which will be the target nodes for each metapath subgraph

            A list of MFGs of per sampled metapath sub-graphs necessary for computing the representation
            of the seed nodes for each metapath sub-graph
        """
        if isinstance(seeds, Mapping):
            assert all([ntype in seeds for ntype in self.metapaths_src_ntype]), \
                "All node types in {} should be present in seeds".format(set(self.metapaths_src_ntype))
        else:
            assert self.num_target_ntypes == 1, \
                "seeds should be dictionary with seeds per node type if the number of target node types is more than 1"
        blocks = []
        for i, metapath_sampler in enumerate(self.sampler_list):
            target_ntype = self.metapaths_src_ntype[i]
            ntype_input_seeds = seeds[target_ntype] if isinstance(seeds, Mapping) else seeds
            if len(metapath_sampler) == 1: ## only one layer of message passing in the computational graph
                frontier = metapath_sampler[0](ntype_input_seeds)
                # add self loop if homogeneous
                if len(frontier.ntypes) == 1:
                    block = dgl.to_block(frontier, ntype_input_seeds)
                else:
                    block = dgl.to_block(frontier, {target_ntype: ntype_input_seeds})
                blocks.append(block)
            else: ## multiple layers of message passing in the computational graph
                layer_blocks = []
                for sampler in metapath_sampler:
                    frontier = sampler(ntype_input_seeds)
                    assert len(frontier.ntypes) == 1, "For multiple layers of message passing"
                    block = dgl.to_block(frontier, ntype_input_seeds)
                    ntype_input_seeds = block.srcdata[dgl.NID]
                    layer_blocks.insert(0, block)
                blocks.append(layer_blocks)
        return blocks

    def sample(self, seeds):
        r"""
        Sample subgraphs for seed nodes

        Parameters
        ----------
        seeds : th.Tensor or dict[ntype: th.Tensor]
            seed nodes of type 'ntype' where ``ntype`` is the ending node type of the given metapath

        Returns
        -------
        list[DGLGraph]

            A list of Subgraph of per sampled metapath
        """
        if isinstance(seeds, Mapping):
            assert all([ntype in seeds for ntype in self.metapaths_src_ntype]), \
                "All node types in {} should be present in seeds".format(set(self.metapaths_src_ntype))
        else:
            assert self.num_target_ntypes == 1, \
                "seeds should be dictionary with seeds per node type if the number of target node types is more than 1"
        frontiers = []
        for i, sampler in enumerate(self.sampler_list):
            target_ntype = self.metapaths_src_ntype[i]
            ntype_input_seeds = seeds[target_ntype] if isinstance(seeds, Mapping) else seeds
            frontiers.append(sampler[0](ntype_input_seeds))

        return frontiers


if __name__ == "__main__":
    from ogb.nodeproppred import DglNodePropPredDataset
    dataset = DglNodePropPredDataset(name='ogbn-mag')
    graph, label = dataset[0]
    train_nids = dataset.get_idx_split()["train"]
    metapath_list = [['cites']]
    sampler = MetapathListSampler(graph, metapath_list, [5, 5])

    # use sampler directly
    random_train_nids = {'paper': np.random.choice(train_nids['paper'], 10)}
    # metapath_subgraphs = sampler.sample(random_train_nids)
    # for metapath, metapath_subg in zip(metapath_list, metapath_subgraphs):
    #     print(metapath)
    #     print(metapath_subg)

    metapath_block_subgraphs = sampler.sample_blocks(random_train_nids)
    for metapath, metapath_block_subg in zip(metapath_list, metapath_block_subgraphs):
        print(metapath)
        print(metapath_block_subg)
