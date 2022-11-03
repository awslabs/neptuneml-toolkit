import argparse
import os
import time
from functools import partial
from neptuneml_toolkit.train import get_training_config, get_train_eids, get_valid_eids
from neptuneml_toolkit.metrics import save_eval_metrics
from neptuneml_toolkit.transform import save_link_prediction_model_artifacts, normalize_hyperparameter_keys
from neptuneml_toolkit.utils import get_device_type
from neptuneml_toolkit.graphloader import GraphLoader
from neptuneml_toolkit.modelzoo import RGCNEncoder, MLPFeatureTransformer, GraphDistmultDecoder
from sklearn.metrics import roc_auc_score

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

def get_model(g, in_size, hyperparameters, model_file=None, device="cpu"):
    model = RGCNLinkPrediction(g.etypes,
                              in_size,
                              int(hyperparameters['hidden_size']),
                              int(hyperparameters['num_bases']),
                              int(hyperparameters['num_encoder_layers']))
    if model_file is not None:
        model_dict = torch.load(model_file, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['model_state_dict'])

    model = model.to(device)
    return model

class RGCNLinkPrediction(nn.Module):
    def __init__(self, etypes, in_sizes, hidden_size, num_bases, num_encoder_layers):
        super(RGCNLinkPrediction, self).__init__()
        self.feature_transformer = MLPFeatureTransformer(in_sizes, hidden_size, per_feat_name=False)
        self.encoder = RGCNEncoder(etypes, hidden_size, hidden_size, num_encoder_layers, num_bases=num_bases)
        self.decoder = GraphDistmultDecoder(hidden_size, len(etypes))

    def forward(self, g, x, pos_graph, neg_graph):
        h = self.feature_transformer(x)
        embeddings = self.encoder(g, h)
        return self.decoder(pos_graph, embeddings), self.decoder(neg_graph, embeddings)

    def get_embeddings(self, g, x,  batch_size, device='cpu', num_workers=0):
        h = self.feature_transformer(x)
        return self.encoder.batch_inference(g, h, batch_size, device=device, num_workers=num_workers)

    def save(self, model_file):
        torch.save({'model_state_dict': self.state_dict()}, model_file)

class RGCNLinkPredictionLightning(LightningModule):
    def __init__(self, graphloader, hyperparameters, device_type='cpu'):
        super().__init__()
        self.graphloader = graphloader
        self.g = graphloader.graph
        print("Loaded graph: {}".format(self.g))

        self.hyperparameters = hyperparameters
        self.features_dict = graphloader.get_node_features()
        input_sizes = {ntype: {feat_name: features[feat_name].shape[1] for feat_name in features}
                       for ntype, features in self.features_dict.items()}
        print("Got input features with shape graph: {}".format(input_sizes))

        self.module = get_model(self.g, input_sizes, hyperparameters, device=device_type)
        print("Created model: {}".format(self.module))

        self.loss_fn = F.binary_cross_entropy_with_logits

        self.val_metric = Accuracy()

    def training_step(self, batch, batch_idx):
        input_nodes, pos_graph, neg_graph, subgraphs = batch
        batch_features = {ntype: {feat_name: feat[node_idx].to(self.device)
                                  for feat_name, feat in self.features_dict[ntype].items()}
                          for ntype, node_idx in input_nodes.items()}
        subgraphs = [subgraph.to(self.device) for subgraph in subgraphs]
        pos_scores, neg_scores = self.module(subgraphs, batch_features, pos_graph.to(self.device), neg_graph.to(self.device))
        pos_scores, neg_scores = torch.cat(list(pos_scores.values())), torch.cat(list(neg_scores.values()))

        predictions = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

        return self.loss_fn(predictions, labels)

    def validation_step(self, batch, batch_idx):
        input_nodes, pos_graph, neg_graph, subgraphs = batch
        batch_features = {ntype: {feat_name: feat[node_idx].to(self.device)
                                  for feat_name, feat in self.features_dict[ntype].items()}
                          for ntype, node_idx in input_nodes.items()}
        pos_scores, neg_scores = self.module(subgraphs, batch_features, pos_graph.to(self.device), neg_graph.to(self.device))
        pos_scores, neg_scores = torch.cat(list(pos_scores.values())), torch.cat(list(neg_scores.values()))

        predictions = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

        self.val_metric((torch.sigmoid(predictions) > 0.5).long(), labels.long())
        self.log('Validation Accuracy', self.val_metric, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters["lr"],
                                  weight_decay=self.hyperparameters["weight_decay"])
        return optimizer

class DataModule(LightningDataModule):

    def __init__(self, graphloader, batch_size, num_layers, num_neighbours=10, num_negs=10, num_workers=0):
        super().__init__()

        self.g = graphloader.graph
        self.train_eids = get_train_eids(self.g)
        self.valid_eids = get_valid_eids(self.g)
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler([num_neighbours] * num_layers)
        self.negative_sampler = dgl.dataloading.negative_sampler.Uniform(num_negs)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return dgl.dataloading.EdgeDataLoader(self.g, self.train_eids, self.sampler, negative_sampler=self.negative_sampler,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              num_workers=0)

    def val_dataloader(self):
        return dgl.dataloading.EdgeDataLoader(self.g, self.valid_eids, self.sampler, negative_sampler=self.negative_sampler,
                                              batch_size=self.batch_size, shuffle=False,
                                              num_workers=0)


def train(data_path, model_path, devices, hyperparameters):
    print("Training config: data_path: {}, model_path: {}, devices: {} hyperparameters: {}".format(data_path,
                                                                                                  model_path,
                                                                                                  devices,
                                                                                                  hyperparameters))
    device_type = get_device_type(devices)
    graphloader = GraphLoader(data_path)

    datamodule = DataModule(graphloader, hyperparameters['batch_size'], hyperparameters['num_encoder_layers'],
                            num_neighbours=hyperparameters['num_neighbors'], num_negs=hyperparameters['num_negs'])

    model = RGCNLinkPredictionLightning(graphloader, hyperparameters, device_type=device_type)

    print("Starting model training")
    checkpoint_callback = ModelCheckpoint(monitor='Validation Accuracy', save_top_k=1)
    trainer = Trainer(default_root_dir=model_path, gpus=1 if device_type == "cuda" else None,
                      max_epochs=hyperparameters['n_epochs'],
                      callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)
    model.module.save(os.path.join(model_path, 'model.pt'))

def transform(data_path, model_path, devices, hyperparameters):
    hyperparameters = normalize_hyperparameter_keys(hyperparameters)
    print("Transform config: data_path: {}, model_path: {}, devices: {} hyperparameters: {}".format(data_path,
                                                                                                    model_path,
                                                                                                    devices,
                                                                                                    hyperparameters))
    device_type = get_device_type(devices)

    graphloader = GraphLoader(data_path)
    g = graphloader.graph

    print("Loaded graph: {}".format(g))

    features_dict = graphloader.get_node_features()
    input_sizes = {ntype: {feat_name: features[feat_name].shape[1] for feat_name in features}
                   for ntype, features in features_dict.items()}
    print("Got input features with shape graph: {}".format(input_sizes))

    model = get_model(g, input_sizes, hyperparameters, device=device_type,
                      model_file=os.path.join(model_path, "model.pt"))
    print("Created model with saved parameters: {}".format(model))

    print("Getting model embeddings")
    node_embeddings = model.get_embeddings(g, features_dict, batch_size=hyperparameters['batch_size'],
                                           device=device_type, num_workers=0)
    relation_type_embeddings = model.decoder.w_relation

    print("Saving model artifacts")
    save_link_prediction_model_artifacts(g, model_path, graphloader, hyperparameters, node_embeddings,
                                         relation_type_embeddings)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action='store_true', default=False, help='Whether script is running locally')
    parser.add_argument("--name", type=str, default='rgcn-link-predict')
    parser.add_argument("--model", type=str, default='custom')
    parser.add_argument("--task", type=str, default='link_predict')
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-negs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default = 128)
    parser.add_argument("--num-bases", type=int, default=2)
    parser.add_argument("--num-encoder-layers", type=int, default=2)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.local:
        data_path, model_path, devices = './lp-tmp', './lp-out', [0]
    else:
        data_path, model_path, devices = get_training_config()

    train(data_path, model_path, devices, vars(args))