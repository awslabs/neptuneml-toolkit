import os
import dgl
import pickle

class GraphLoader(object):
    r""" Load DGLGraph features and metadata created by Neptune ML DataProcessing Job.

    Parameters
    ----------
    path : str
        path where graph binary and metadata are stored
    name : str, optional
        name of the graph loader

    >>> graph_loader = neptuneml.GraphLoader()

    ** Get the graph **
    >>> g = graphloader.graph

    ** Get the node features **
     >>> g = graphloader.get_node_features()

    ** Get the edge features **
     >>> g = graphloader.get_edge_features()
    """
    def __init__(self, path, name='graph'):
        self._name = name
        self._graph = None
        self._node_dict = {}
        self._label_map = None
        self._is_multilabel = False
        self._label_properties = None
        self._node_feat_meta = None
        self._edge_feat_meta = None
        self.load(path)

    def load(self, path):
        """load the graph and the labels

        Parameters
        ----------
        path: str
            Path where to load the graph and the labels
        """
        graph_path = os.path.join(path, 'graph.bin')
        info_path = os.path.join(path, 'info.pkl')
        graphs, _ = dgl.data.load_graphs(graph_path)
        self._g = graphs[0]

        with open(info_path, "rb") as pf:
            info = pickle.load(pf)
        self._node_dict = info['node_id_map']
        self._label_map = info['label_map']
        self._label_properties = info['label_properties']
        self._is_multilabel = info['is_multilabel']
        self._node_feat_meta = info['node_feat_meta']
        self._edge_feat_meta = info['edge_feat_meta']

    def get_node_features(self):
        assert self._node_feat_meta is not None and self._g is not None,\
            "Graph has not been loaded. you must load graph before getting node features. run `graphloader.load()`"
        nf = {ntype: {} for ntype in self._node_feat_meta}
        for ntype in self._node_feat_meta:
            for feature in self._node_feat_meta[ntype]:
                if self._node_feat_meta[ntype][feature]['type'] == 'raw_feat':
                    nf[ntype][feature] = self._node_feat_meta[ntype][feature]['raw_feat']
                else:
                    nf[ntype][feature] = self._g.nodes[ntype].data[feature]
        return nf

    def get_edge_features(self):
        assert self._edge_feat_meta is not None and self._g is not None, \
            "Graph has not been loaded. you must load graph before getting edge features. run `graphloader.load()`"
        ef = {etype: {} for etype in self._edge_feat_meta}
        for etype in self._edge_feat_meta:
            for feature in self._edge_feat_meta[etype]:
                if self._edge_feat_meta[etype][feature]['type'] == 'raw_feat':
                    ef[etype][feature] = self._edge_feat_meta[etype][feature]['raw_feat']
                else:
                    ef[etype][feature] = self._g.edges[etype].data[feature]
        return ef

    @property
    def rtype2id(self):
        """ Return mappings from relation type to internal node id

        Note: used only for KGE models

        Return
        ------
        dict of dict:
            {relation_type : {raw relation id(string/int): dgl_id}}
        """
        return self._rel_dict

    @property
    def node2id(self):
        """ Return mappings from raw node id/name to internal node id

        Return
        ------
        dict of dict:
            {node_type : {raw node id(string/int): dgl_id}}
        """
        return self._node_dict

    @property
    def id2node(self):
        """ Return mappings from internal node id to raw node id/name

        Return
        ------
        dict of dict:
            {node_type : {raw node id(string/int): dgl_id}}
        """
        return {node_type: {val: key for key, val in node_maps.items()} \
                for node_type, node_maps in self._node_dict.items()}

    @property
    def label_map(self):
        """ Return mapping from internal label id to original label

        Return
        ------
        dict:
            {type: {label id(int) : raw label(string/int)}}
        """
        return self._label_map

    @property
    def label_properties(self):
        """ Return mapping from node/edge type to label column name or property

        Return
        ------
        dict:
            {type: label_property (string/int)}
        """
        return self._label_properties

    @property
    def graph(self):
        """ Return graph
        """
        return self._g

    @property
    def is_multilabel(self):
        """ Return whether label is multilabel or singlelebel
        """
        return self._is_multilabel

    @property
    def edge_feat_meta(self):
        return self._edge_feat_meta

    @property
    def node_feat_meta(self):
        return self._node_feat_meta
