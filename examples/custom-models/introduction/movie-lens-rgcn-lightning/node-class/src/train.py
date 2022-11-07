import argparse
import os
import time
from functools import partial
from neptuneml_toolkit.train import get_training_config, get_train_nids, get_valid_nids, get_node_labels
from neptuneml_toolkit.metrics import classification_report, save_eval_metrics
from neptuneml_toolkit.transform import save_node_prediction_model_artifacts, normalize_hyperparameter_keys
from neptuneml_toolkit.utils import get_device_type
from neptuneml_toolkit.graphloader import GraphLoader
from neptuneml_toolkit.modelzoo import RGCNEncoder, MLP, MLPFeatureTransformer
from sklearn.metrics import roc_auc_score

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

def get_model(g, in_size, out_size, hyperparameters, model_file=None, device="cpu"):
    model = RGCNNodeClassification(g.etypes,
                                  hyperparameters['target_ntype'],
                                  in_size,
                                  int(hyperparameters['hidden_size']),
                                  int(hyperparameters['num_bases']),
                                  out_size,
                                  int(hyperparameters['num_encoder_layers']),
                                  int(hyperparameters['num_decoder_layers']))
    if model_file is not None:
        model_dict = torch.load(model_file, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['model_state_dict'])

    model = model.to(device)
    return model

class RGCNNodeClassification(nn.Module):
    def __init__(self, etypes, target_ntype, in_sizes, hidden_size, num_bases, out_size, num_encoder_layers, num_decoder_layers):
        super(RGCNNodeClassification, self).__init__()
        self.target_ntype = target_ntype
        self.feature_transformer = MLPFeatureTransformer(in_sizes, hidden_size, per_feat_name=False)
        self.encoder = RGCNEncoder(etypes, hidden_size, hidden_size, num_encoder_layers, num_bases=num_bases)
        self.decoder = MLP(hidden_size, out_size, num_layers=num_decoder_layers)

    def forward(self, g, x):
        h = self.feature_transformer(x)
        embeddings = self.encoder(g, h)
        return self.decoder(embeddings[self.target_ntype])

    def get_embeddings(self, g, x,  batch_size, device='cpu', num_workers=0):
        h = self.feature_transformer(x)
        return self.encoder.batch_inference(g, h, batch_size, device=device, num_workers=num_workers)

    def save(self, model_file):
        torch.save({'model_state_dict': self.state_dict()}, model_file)

class RGCNNodeClassificationLightning(LightningModule):
    def __init__(self, graphloader, hyperparameters, device_type='cpu'):
        super().__init__()
        self.graphloader = graphloader
        self.g = graphloader.graph
        print("Loaded graph: {}".format(self.g))

        self.hyperparameters = hyperparameters
        self.target_ntype = hyperparameters['target_ntype']
        self.features_dict = graphloader.get_node_features()
        input_sizes = {ntype: {feat_name: features[feat_name].shape[1] for feat_name in features}
                       for ntype, features in self.features_dict.items()}
        print("Got input features with shape graph: {}".format(input_sizes))

        self.labels = get_node_labels(self.g, self.target_ntype)
        output_size = len(graphloader.label_map[self.target_ntype])
        print("Got training labels with shape graph: {}".format(self.labels.shape))

        self.module = get_model(self.g, input_sizes, output_size, hyperparameters, device=device_type)
        print("Created model: {}".format(self.module))

        self.loss_fn = F.binary_cross_entropy_with_logits

        self.train_metric = Accuracy(num_classes=self.labels.shape[1], average='weighted')
        self.val_metric = Accuracy(num_classes=self.labels.shape[1], average='weighted')

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, subgraphs = batch
        batch_features = {ntype: {feat_name: feat[node_idx].to(self.device)
                                  for feat_name, feat in self.features_dict[ntype].items()}
                          for ntype, node_idx in input_nodes.items()}
        subgraphs = [subgraph.to(self.device) for subgraph in subgraphs]
        logits = self.module(subgraphs, batch_features)

        batch_labels = self.labels[output_nodes[self.target_ntype]].to(self.device)
        loss = self.loss_fn(logits, batch_labels.float())
        self.train_metric((torch.sigmoid(logits) > 0.5).long(), batch_labels)
        self.log('Train Accuracy', self.train_metric, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, subgraphs = batch
        batch_features = {ntype: {feat_name: feat[node_idx].to(self.device)
                                  for feat_name, feat in self.features_dict[ntype].items()}
                          for ntype, node_idx in input_nodes.items()}
        subgraphs = [subgraph.to(self.device) for subgraph in subgraphs]
        logits = self.module(subgraphs, batch_features)

        batch_labels = self.labels[output_nodes[self.target_ntype]].to(self.device)
        self.val_metric((torch.sigmoid(logits) > 0.5).long(), batch_labels)
        self.log('Validation Accuracy', self.val_metric, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters["lr"],
                                  weight_decay=self.hyperparameters["weight_decay"])
        return optimizer

class DataModule(LightningDataModule):

    def __init__(self, graphloader, target_ntype, batch_size, num_layers, num_neighbours=10, num_workers=0):
        super().__init__()

        self.g = graphloader.graph
        self.train_nids = {target_ntype: get_train_nids(self.g, target_ntype)}
        self.valid_nids = {target_ntype: get_valid_nids(self.g, target_ntype)}
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler([num_neighbours] * num_layers)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return dgl.dataloading.NodeDataLoader(self.g, self.train_nids, self.sampler,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              num_workers=0)

    def val_dataloader(self):
        return dgl.dataloading.NodeDataLoader(self.g, self.valid_nids, self.sampler,
                                              batch_size=self.batch_size, shuffle=False,
                                              num_workers=0)


def train(data_path, model_path, devices, hyperparameters):
    print("Training config: data_path: {}, model_path: {}, devices: {} hyperparameters: {}".format(data_path,
                                                                                                  model_path,
                                                                                                  devices,
                                                                                                  hyperparameters))
    device_type = get_device_type(devices)
    graphloader = GraphLoader(data_path)

    datamodule = DataModule(graphloader, hyperparameters['target_ntype'], hyperparameters['batch_size'],
                          hyperparameters['num_encoder_layers'], num_neighbours=hyperparameters['num_neighbors'])

    model = RGCNNodeClassificationLightning(graphloader, hyperparameters, device_type=device_type)

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

    features_dict = graphloader.get_node_features()
    input_sizes = {ntype: {feat_name: features[feat_name].shape[1] for feat_name in features}
                   for ntype, features in features_dict.items()}
    output_size =  len(graphloader.label_map[hyperparameters["target_ntype"]])
    print("Got graph input features with shape: {}".format(input_sizes))

    model = get_model(graphloader.graph, input_sizes, output_size, hyperparameters, device=device_type,
                      model_file=os.path.join(model_path, "model.pt"))

    node_embeddings = model.get_embeddings(graphloader.graph, features_dict, batch_size=hyperparameters['batch_size'],
                                           device=device_type, num_workers=0)
    predictions = torch.sigmoid(model.decoder(node_embeddings[hyperparameters["target_ntype"]]))

    print("Saving model artifacts")
    save_node_prediction_model_artifacts(model_path, predictions, graphloader, hyperparameters, embeddings=node_embeddings)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action='store_true', default=False, help='Whether script is running locally')
    parser.add_argument("--name", type=str, default='rgcn-node-class')
    parser.add_argument("--model", type=str, default='custom')
    parser.add_argument("--task", type=str, default='node_class')
    parser.add_argument("--property", type=str, default='label')
    parser.add_argument("--target_ntype", type=str, default='movie')
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default = 128)
    parser.add_argument("--num-bases", type=int, default=2)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=1)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.local:
        data_path, model_path, devices = './nc-tmp', './nc-out', [0]
    else:
        data_path, model_path, devices = get_training_config()

    train(data_path, model_path, devices, vars(args))