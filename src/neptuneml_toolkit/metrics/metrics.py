import os
import json
import sklearn
import tqdm
import torch
from collections.abc import Mapping

def classification_report(labels, predictions, return_dict=True, label_map=None, label_indices=None, target_names=None):
    r"""

    Print classification report and return as a dict

    Parameters
    ----------
    labels : th.Tensor
        1 d torch tensor with true class labels. entries must be [0, n_classes-1]
    predictions : th.Tensor
        1 d torch tensor with same shape. entries must be [0, n_classes-1]
    return_dict : boolean default=True
       whether to also return output as dict
    label_map : dict[int, str] default=None
        mapping from label indices to label class names. overrides label_indices and target_names
    label_indices: list[int] default=None
        Optional list of label indices to include in the report.
    target_names: list[str] default=None
        Optional display names matching the labels indices (same order).
    Returns
    -------
    report: dict
    """

    if label_map is not None:
        label_indices, target_names = zip(sorted(label_map))
    l =  labels.cpu().numpy() if torch.is_tensor(labels) else labels
    p = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else predictions
    print(sklearn.metrics.classification_report(l, p, zero_division=0, labels=label_indices, target_names=target_names))
    if return_dict:
        return sklearn.metrics.classification_report(l, p, output_dict=True, zero_division=0, labels=label_indices,
                                                     target_names=target_names)


def save_eval_metrics(metrics, path):
    r"""

    Save evaluation metrics

    Parameters
    ----------
    metric : dict
        dict container with metric info
    path : str
        directory path to save to
    """
    with open(os.path.join(path, "eval_metrics_info.json"), "w") as f:
        json.dump(metrics, f)


def mrr(node_embeddings, query_edges, scoring_fn, masked_edges=None, src_type=None, dst_type=None, hits=[1]):
    r"""

    Compute and return MRR (mean reciprocal rank) metric for query edges using node embeddings and scoring_fn

    See https://en.wikipedia.org/wiki/Mean_reciprocal_rank for details.


    Parameters
    ----------
    node_embeddings : dict[ntype:str, embed:th.Tensor] or th.Tensor
        node embeddings per node type to be used for computing edge scores
        if th.Tensor and not dictionary then graph must have one node type
    query_edges: tuple(u:th.Tensor, v:th.Tensor, etype_ids:th.Tensor)
        the edges to compute to perturb for computing the mrr in the format (u, v, etype_ids)
        where u is a Tensor of src_node ids, v is a Tensor of dst_node ids and etype_ids is a tensor of edge type ids
    scoring_fn : Callable
        a function that takes embed_u, embed_v and etype_ids and return the score
    masked_edges: tuple(u:th.Tensor, v:th.Tensor, etyped_id:th.Tensor) default=None
        Optional list of edges to exclude when ranking hypothetical edges against target query edge.
    src_type: str default=None
        Optional node type of src nodes in query_edges. Required if node_embeddings is a mapping
    dst_type: str default=None
        Optional node type of dst nodes in query_edges. Required if node_embeddings is a mapping
    hits: list[int] default=[1]
        list of k to return Hits@k
    """
    if isinstance(node_embeddings, Mapping):
        assert src_type is not None and dst_type is not None, "src_type and dst_type must be passed if node_embeddings is a dictionary"
    with torch.no_grad():
        if masked_edges is None:
            triplets_to_filter = []
        else:
            triplets_to_filter = torch.cat([t.reshape(-1, 1) for t in masked_edges], dim=1).tolist()
            triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

        print('Perturbing src nodes...')
        # iterate each query edge and get the rank for all potential src nodes
        ranks_s = _perturb_src_and_get_filtered_rank(node_embeddings, scoring_fn, query_edges, src_type, dst_type, triplets_to_filter)
        print('Perturbing dst nodes...')
        # iterate each query edge and get the rank for all potential dst nodes
        ranks_d = _perturb_dst_and_get_filtered_rank(node_embeddings, scoring_fn, query_edges, src_type, dst_type, triplets_to_filter)

        ranks = torch.cat([ranks_s, ranks_d])
        ranks += 1  # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        hits_at = []
        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            hits_at.append(avg_count.item())
    return mrr.item(), hits_at

def _perturb_src_and_get_filtered_rank(embedding, scoring_fn, query_edges, src_type=None, dst_type=None, triplets_to_filter=[]):
    num_entities = embedding[src_type].shape[0] if src_type is not None else embedding.shape[0]
    ranks = []
    for target_src, target_dst, target_rel in tqdm.tqdm(zip(*query_edges)):
        target_src, target_dst, target_rel = int(target_src), int(target_dst), int(target_rel)
        src_queries = _get_perturbed_src_queries(triplets_to_filter, target_src, target_rel, target_dst, num_entities)
        emb_s = embedding[src_type][src_queries] if src_type is not None else embedding[src_queries]
        emb_d = embedding[dst_type][target_dst] if dst_type is not None else embedding[target_dst]
        emb_d = emb_d.repeat(emb_s.shape[0], 1)
        rel_ids = torch.full((emb_d.shape[0],), target_rel, dtype=torch.long)
        scores = scoring_fn(emb_s, emb_d, rel_ids)
        _, indices = torch.sort(scores, descending=True)
        target_src_idx = int((src_queries == target_src).nonzero()) # index of current target src
        rank = int((indices == target_src_idx).nonzero()) # rank by score of current target
        ranks.append(rank)
    return torch.LongTensor(ranks)

def _perturb_dst_and_get_filtered_rank(embedding, scoring_fn, query_edges, src_type=None, dst_type=None, triplets_to_filter=[]):
    num_entities = embedding[dst_type].shape[0] if dst_type is not None else embedding.shape[0]
    ranks = []
    for target_src, target_dst, target_rel in tqdm.tqdm(zip(*query_edges)):
        target_src, target_dst, target_rel = int(target_src), int(target_dst), int(target_rel)
        dst_queries = _get_perturbed_dst_queries(triplets_to_filter, target_src, target_rel, target_dst, num_entities)
        emb_d = embedding[dst_type][dst_queries] if dst_type is not None else embedding[dst_queries]
        emb_s = embedding[src_type][target_src] if src_type is not None else embedding[target_src]
        emb_s = emb_s.repeat(emb_d.shape[0], 1)
        rel_ids = torch.full((emb_d.shape[0],), target_rel, dtype=torch.long)
        scores = scoring_fn(emb_s, emb_d, rel_ids)
        _, indices = torch.sort(scores, descending=True)
        target_dst_idx = int((dst_queries == target_dst).nonzero())  # index of current target dst
        rank = int((indices == target_dst_idx).nonzero())  # rank by score of current target
        ranks.append(rank)
    return torch.LongTensor(ranks)

def _get_perturbed_src_queries(triplets_to_filter, target_src, target_rel, target_dst, num_entities):
    return torch.LongTensor([s for s in range(num_entities)
                             if s == target_src or (s, target_dst, target_rel) not in triplets_to_filter])


def _get_perturbed_dst_queries(triplets_to_filter, target_src, target_rel, target_dst, num_entities):
    return torch.LongTensor([d for d in range(num_entities)
                             if d == target_dst or (target_src, d, target_rel) not in triplets_to_filter])