import os
import json
import torch as th
from collections.abc import Mapping
from ..utils import get_available_devices

def get_training_config():
    r"""

    Returns
    -------
    tuple(data_path: str, model_path: str, devices: list[int])
    """
    data_path = str(os.environ['SM_CHANNEL_TRAIN'])
    model_path = '/opt/ml/checkpoints'
    devices = get_available_devices()

    return data_path, model_path, devices

def get_train_nids(g, target_ntype=None):
    r"""

    get validation node ids from the graph

    Parameters
    ----------
    g : dgl.DGLGraph
        the Graph
    target_ntype : str
        the node type of the targets
    Returns
    -------
    train_idx: th.Tensor
    """
    train_mask = g.ndata['train_mask'] if target_ntype is None else g.nodes[target_ntype].data['train_mask']
    train_idx = th.arange(train_mask.shape[0])
    train_idx = train_idx[train_mask.bool()]
    return train_idx

def get_valid_nids(g, target_ntype=None):
    r"""

    get validation node ids from the graph

    Parameters
    ----------
    g : dgl.DGLGraph
        the Graph
    target_ntype : str
        the node type of the targets
    Returns
    -------
    valid_idx: th.Tensor
    """
    valid_mask = g.ndata['valid_mask'].bool() if target_ntype is None else g.nodes[target_ntype].data['valid_mask']
    valid_idx = th.arange(valid_mask.shape[0])
    valid_idx = valid_idx[valid_mask.bool()]
    return valid_idx

def get_node_labels(g, target_ntype=None):
    r"""

    get node labels from the graph

    Parameters
    ----------
    g : dgl.DGLGraph
        the Graph
    target_ntype : str (optional)
        the node type of the targets
   Returns
   -------
   labels: th.Tensor
   """
    labels = g.ndata['labels'] if target_ntype is None else g.nodes[target_ntype].data['labels']
    return labels

def get_train_eids(g, target_etype=None, include_reverse_etypes=True):
    r"""

    get validation node ids from the graph

    Parameters
    ----------
    g : dgl.DGLGraph
        the Graph
    target_etype : str
        the edge type of the targets
    include_reverse_etypes : bool (default: True)
        whether to include edge ids from reversed edge types.
    Returns
    -------
    train_idx: th.Tensor
    """
    train_mask = {target_etype: g.edges[target_etype].data['train_mask']} if target_etype  else g.edata['train_mask']
    if include_reverse_etypes and isinstance(train_mask, Mapping):
        for etype in train_mask:
            src_type, rel_type, dst_type = etype
            if "rev-" in rel_type and (dst_type, rel_type.replace("rev-", ""), src_type) in train_mask:
                train_mask[etype] = g.edata['rev_train_mask'][etype]
    if isinstance(train_mask, Mapping):
        train_idx = {}
        for etype in train_mask:
            idx = th.arange(train_mask[etype].shape[0])
            train_idx[etype] = idx[train_mask[etype].bool()]
    else:
        train_idx = th.arange(train_mask.shape[0])
        train_idx = train_idx[train_mask.bool()]
    return train_idx

def get_valid_eids(g, target_etype=None, include_reverse_etypes=False):
    r"""

    get validation node ids from the graph

    Parameters
    ----------
    g : dgl.DGLGraph
        the Graph
    target_etype : str
        the node type of the targets
    include_reverse_etypes : bool (default: False)
        whether to include edge ids from reversed edge types.
    Returns
    -------
    valid_idx: th.Tensor or dict[etype:str : th.Tensor]
    """
    valid_mask = {target_etype: g.edges[target_etype].data['valid_mask']} if target_etype  else g.edata['valid_mask']
    if include_reverse_etypes and isinstance(train_mask, Mapping):
        for etype in valid_mask:
            src_type, rel_type, dst_type = etype
            if "rev-" in rel_type and (dst_type, rel_type.replace("rev-", ""), src_type) in valid_mask:
                valid_mask[etype] = g.edata['rev_valid_mask'][etype]
    if isinstance(valid_mask, Mapping):
        valid_idx = {}
        for etype in valid_mask:
            idx = th.arange(valid_mask[etype].shape[0])
            valid_idx[etype] = idx[valid_mask[etype].bool()]
    else:
        valid_idx = th.arange(valid_mask.shape[0])
        valid_idx = valid_idx[valid_mask.bool()]
    return valid_idx

def get_edge_labels(g, target_etype=None):
    r"""

    get node labels from the graph

    Parameters
    ----------
    g : dgl.DGLGraph
        the Graph
    target_etype : str (optional)
        the edge type of the targets
   Returns
   -------
   labels: th.Tensor
   """
    labels = g.edata['labels'] if target_etype is None else g.edges[target_etype].data['labels']
    return labels

def get_all_forward_edges(g):
    r"""

    get all forward (non-reversed) edges from the graph in the u, v, etype_id

    Parameters
    ----------
    g : dgl.DGLGraph
        the Graph

   Returns
   -------
   labels: (u: th.Tensor, v: th.Tensor, etype_ids: th.Tensor)
   """
    srcs, dsts, etype_ids = [], [], []
    for etype_id, etype in enumerate(g.canonical_etypes):
        if "rev-" in etype[1]:
            continue
        u, v = g.edges(etype=etype)
        srcs.append(u)
        dsts.append(v)
        etype_ids.append(th.full_like(u, etype_id))

    return th.cat(srcs), th.cat(dsts), th.cat(etype_ids)
