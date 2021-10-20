import dgl
import torch as th
import torch.nn as nn

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class NodeEmbedding(nn.Module):
    r"""
    Learnable node embeddings as initial node features

    Parameters
    ----------
    node_counts : dict[ntype: str, ncount: int]
        list of edge types/relations in the graph
    embed_size : int
        Size of embedding vector for each node
    use_torch: bool [Optional]
        whether to use torch embeddings or dgl Node Embeddings (Default: False)
    init_missing_embed_with: str [Optional]
        when loading from saved embeddings how to initialize embedding vectors for missing nodes (Default: 'random')
        options: ['random', 'average', 'zero']
    dev_id: int [Optional]
        device_id used in multigpu training (Default: 0)
    activation: Callable
        callable activation function/layer or None, optional. (Default: torch.nn.Functional.relu)
    """

    def __init__(self, node_counts, embed_size, use_torch=False, init_missing_embed_with='random', dev_id=0):
        super(NodeEmbedding, self).__init__()

        self.embed_size = embed_size
        self.use_torch_emb = use_torch
        self.init_missing_embed_with = init_missing_embed_with
        self.dev_id = dev_id
        self.node_embeds = nn.ModuleDict() if self.use_torch_emb else {}
        for ntype, ncount in node_counts.items():
            if self.use_torch_emb:
                self.node_embeds[str(ntype)] = nn.Embedding(ncount, self.embed_size)
            else:
                self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(ncount, self.embed_size, name=str(ntype),
                                                                    init_func=initializer)


    def load_node_embeds(self, embs):
        for key, emb in embs.items():
            if self.use_torch_emb:
                if  self.node_embeds[key].weight.shape[0] == emb.shape[0]:
                    self.node_embeds[key] = nn.Embedding.from_pretrained(emb, freeze=True)
                else:
                    print("Initializing node embeddings for previously unseen nodes.")
                    if self.init_missing_embed_with == 'zero':
                        self.node_embeds[key].weight.data = nn.Parameter(th.zeros((self.node_embeds[key].weight.shape)))
                    elif self.init_missing_embed_with== 'average':
                        self.node_embeds[key].weight.data = nn.Parameter(th.mean(emb,dim=0,keepdim=True)
                                                                         .repeat(self.node_embeds[key].weight.shape[0],1))
                    self.node_embeds[key].weight.data[:emb.shape[0]] = nn.Parameter(emb[:])
            else:
                if self.node_embeds[key].emb_tensor.shape[0] == emb.shape[0]:
                    self.node_embeds[key].emb_tensor[:] = emb[:]
                else:
                    print("Initializing node embeddings for previously unseen nodes.")
                    if self.init_missing_embed_with == 'zero':
                        self.node_embeds[key].emb_tensor.data = nn.Parameter(th.zeros(self.node_embeds[key].emb_tensor.shape))
                    elif self.init_missing_embed_with == 'average':
                        self.node_embeds[key].emb_tensor.data = nn.Parameter(th.mean(emb,dim=0,keepdim=True).
                                                                        repeat(self.node_embeds[key].emb_tensor.shape[0],1))
                    self.node_embeds[key].emb_tensor.data[:emb.shape[0]]  = nn.Parameter(emb[:])

    def forward(self, node_ids):
        """Forward computation
        Parameters
        ----------
        node_ids : dict[ntype: str : ntype_nids: Tensor]
            node ids to generate embedding for.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """

        embeds = {}
        for ntype in node_ids:
            if node_ids[str(ntype)].shape[0] == 0:
                continue
            if self.use_torch_emb:
                embeds[str(ntype)] = self.node_embeds[str(ntype)](node_ids[str(ntype)])
            else:
                embeds[str(ntype)] = self.node_embeds[str(ntype)](node_ids[str(ntype)], self.dev_id)
        return embeds

    @property
    def dgl_emb(self):
        """
        """
        if self.use_torch_emb:
            return None
        embs = [emb for emb in self.node_embeds.values()]
        return embs