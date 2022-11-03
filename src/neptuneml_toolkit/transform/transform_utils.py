import os
import json
import pickle
import numpy as np
import torch as th
import dgl
from collections.abc import Mapping
from ..utils import get_available_devices

TUNING_JOB_DETAIILS_FILE = 'tuning_job_details.json'
BEST_TRAINING_JOB_KEY = 'BestTrainingJob'
TRAINING_HYPERPARAMETERS_KEY = 'TrainingJobParameters'
TRAINING_JOB_DETAIILS_FILE = 'training_job_details.json'

def get_transform_config():
    r"""

    Returns
    -------
    tuple(data_path: str, model_path: str, devices: list[int], hyperparameters: dict[str, str])
    """
    data_path = os.environ['SM_DATA_DIR']
    output_path = os.environ['SM_OUTPUT_DIR']
    model_path = os.environ['SM_MODEL_DIR']
    devices = get_available_devices()

    if TUNING_JOB_DETAIILS_FILE in os.listdir(output_path):
        tuning_job_details = os.path.join(output_path, TUNING_JOB_DETAIILS_FILE)
        with open(tuning_job_details, 'r') as f:
            params = json.load(f)[BEST_TRAINING_JOB_KEY][TRAINING_HYPERPARAMETERS_KEY]

    else:
        training_job_details = os.path.join(data_path, TRAINING_JOB_DETAIILS_FILE)
        with open(training_job_details, 'r') as f:
            params = json.load(f)[TRAINING_HYPERPARAMETERS_KEY]

    return data_path, model_path, devices, params

def save_node_prediction_model_artifacts(model_path, predictions, graphloader, params, embeddings=None):
    r"""

    save model artifacts necessary for inference for Neptune ML node prediction task

    Parameters
    ----------
    model_path : str
        model directory to save artifacts
    predictions : th.Tensor
        model predictions for all nodes
    graphloader : neptuneml.GraphLoader
        the GraphLoader object
    params : dict[str: str]
        hyperparameter dictionary for the task
    embeddings : th.Tensor (optional)
        computed node embeddings. optional for node prediction tasks
    """
    if th.is_tensor(predictions):
        predictions= predictions.detach().cpu().numpy()
    np.savez(os.path.join(model_path, 'result.npz'), infer_scores=predictions)
    if embeddings is not None:
        if isinstance(embeddings, Mapping):
            save_entity_embeddings(embeddings, model_path, target_ntype=params['target_ntype'])
        else:
            save_entity_embeddings(embeddings, model_path)
    _save_node_prediction_model_metadata(graphloader, model_path, params['target_ntype'])
    _save_inference_config(model_path, params)

def _save_node_prediction_model_metadata(graph_data, path, target_ntype):
    with open(os.path.join(path, 'mapping.info'), "wb") as f:
        node2id = graph_data.node2id
        label_map = graph_data.label_map
        pickle.dump({'node2id': node2id,
                     'label_map': label_map,
                     'target_ntype': target_ntype,
                     'target_ntype_property': graph_data.label_properties[target_ntype]}, f)

def save_link_prediction_model_artifacts(g, model_path, graphloader, params, entity_embeddings, rel_type_embeddings):
    r"""

    save model artifacts necessary for inference for Neptune ML node prediction task

    Parameters
    ----------
    g: dgl.DGLGraph
        the dgl graph
    model_path : str
        model directory to save artifacts
    graphloader : neptuneml.GraphLoader
        the GraphLoader object
    params : dict[str: str]
        hyperparameter dictionary for the task
    entity_embeddings : dict[ntype:str: th.Tensor]
        computed node embeddings.
    rel_type_embeddings : th.Tensor
        learned relation embeddings.
    """
    _save_link_prediction_model_metadata(g, graphloader, entity_embeddings, rel_type_embeddings, params, model_path)
    _save_inference_config(model_path, params)

def _save_link_prediction_model_metadata(g, graph_data, entity_embeddings, rel_type_embeddings, params, path):
    non_reverse_edges = [etype for etype in g.canonical_etypes if 'rev-' not in etype[1]]
    non_reverse_etype_ids = [etid for etid, etype in enumerate(g.canonical_etypes) if etype in non_reverse_edges]
    non_reverse_etype_embeddings = rel_type_embeddings[non_reverse_etype_ids, :]
    if th.is_tensor(non_reverse_etype_embeddings):
        non_reverse_etype_embeddings = non_reverse_etype_embeddings.detach().cpu().numpy()

    eg = dgl.edge_type_subgraph(g, non_reverse_edges)
    et2id_map = {e_type: etid for etid, e_type in enumerate(eg.canonical_etypes)}
    nt2id_map = {n_type: ntid for ntid, n_type in enumerate(eg.ntypes)}

    g = dgl.to_homogeneous(eg)
    localid2globalid = {ntype: th.nonzero(g.ndata[dgl.NTYPE] == ntid).squeeze().numpy()
                        for ntype, ntid in nt2id_map.items()}

    with open(os.path.join(path, 'mapping.info'), "wb") as f:
        node2id = graph_data.node2id
        label_map = graph_data.label_map
        pickle.dump({'model_type': params['model'],
                     'node2id': node2id,
                     'label_map': label_map,
                     'relation2id_map': et2id_map,
                     'nodetype2id_map': nt2id_map,
                     'node2gid': localid2globalid}, f)

    dgl.data.save_graphs(os.path.join(path, 'graph.bin'), g)
    np.save(os.path.join(path, 'relation.npy'), non_reverse_etype_embeddings)
    save_entity_embeddings(entity_embeddings, path, tnid2gnid=localid2globalid)

def save_entity_embeddings(entity_embeddings, path, target_ntype=None, tnid2gnid=None):
    r"""

    save entity embeddings for model inference

    Parameters
    ----------
    entity_embeddings : th.Tensor or dict[str: th.Tensor]
        entity embeddings for a single node type or per node type
    path : str
        model directory to save embeddings
    target_ntype : str (optional)
        the node type of the target.
        if entity_embeddings is a dict causes embedding tensor corresponding to this ntype only to be saved
    tnid2gnid : dict[ntype:str,  np.array[gnid:int]] (optional)
        mapping for node type specific node index to global node index.
        required if entity_embeddings is a dictionary and target_ntype is not set
        since all the embeddings for each node type are concatenated into a single embedding tensor for all nodes
    """
    filename = os.path.join(path, 'entity.npy')
    if isinstance(entity_embeddings, Mapping):
        if target_ntype:
            emb = entity_embeddings[target_ntype]
            if th.is_tensor(emb):
                emb = emb.detach().cpu().numpy()
            np.save(filename, emb)
        else:
            assert tnid2gnid is not None, "Must pass mapping node type node ids to global node ids"
            embed_dim = [embed.shape[1] for embed in entity_embeddings.values()][0]
            total_nodes = sum([embed.shape[0] for embed in entity_embeddings.values()])
            embeddings = np.zeros((total_nodes, embed_dim))
            for ntype, embed in entity_embeddings.items():
                if th.is_tensor(embed):
                    embed = embed.detach().cpu().numpy()
                embeddings[tnid2gnid[ntype], :] = embed
            np.save(filename, embeddings)
    else:
        emb = entity_embeddings.detach().cpu().numpy() if th.is_tensor(entity_embeddings) else entity_embeddings
        np.save(filename, emb)

def _save_inference_config(model_path, params):
    ## generate the inference_config dir and save inference config
    inference_config = {"task": params['task'], "model": params['model'], "name": params['name'],
                        "infer_type": "online-transductive-infer"}
    if 'property' in params:
        inference_config['property'] = params['property']
    inference_config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(inference_config_path):
        with open(inference_config_path, 'r') as f:
            inference_config.update(json.load(f))
    with open(inference_config_path, 'w') as f:
        json.dump(inference_config, f)

def get_test_nids(g, target_ntype=None):
    r"""

    get test node ids from the graph

    Parameters
    ----------
    g : dgl.DGLGraph
        the Graph
    target_ntype : str
        the node type of the targets
   Returns
   -------
   test_idx: th.Tensor
   """
    if 'test_mask' not in g.ndata:
        return None
    test_mask = g.ndata['test_mask'].bool() if target_ntype is None else g.nodes[target_ntype].data['test_mask']
    test_idx = th.arange(test_mask.shape[0])
    test_idx = test_idx[test_mask.bool()]
    return test_idx

def get_test_eids(g, target_etype=None, include_reverse_etypes=False):
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
    test_idx: th.Tensor or dict[etype:str : th.Tensor]
    """
    test_mask = {target_etype: g.edges[target_etype].data['test_mask']} if target_etype  else g.edata['test_mask']
    if include_reverse_etypes and isinstance(train_mask, Mapping):
        for etype in test_mask:
            src_type, rel_type, dst_type = etype
            if "rev-" in rel_type and (dst_type, rel_type.replace("rev-", ""), src_type) in test_mask:
                test_mask[etype] = g.edata['rev_test_mask'][etype]
    if isinstance(test_mask, Mapping):
        test_idx = {}
        for etype in test_mask:
            idx = th.arange(test_mask[etype].shape[0])
            test_idx[etype] = idx[test_mask[etype].bool()]
    else:
        test_idx = th.arange(test_mask.shape[0])
        test_idx = test_idx[test_mask.bool()]
    return test_idx


def normalize_hyperparameter_keys(params):
    r"""

    format the hyperparameter keys so that the names can be

    Parameters
    ----------
    params : dict[str: str]
        the hyperparameters
   Returns
   -------
   dict[str: str]
   """
    new_params = {}
    for key, value in params.items():
        if "-" in key:
            new_key = key.replace("-", "_")
            new_params[new_key] = value
        else:
            new_params[key] = value

    return new_params

