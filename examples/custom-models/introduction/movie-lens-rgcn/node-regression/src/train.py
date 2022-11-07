import argparse
import os
import time
from functools import partial
from neptuneml_toolkit.train import get_training_config, get_train_nids, get_valid_nids, get_node_labels
from neptuneml_toolkit.metrics import save_eval_metrics
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

class RGCNNodeRegression(nn.Module):
    def __init__(self, etypes, target_ntype, in_sizes, hidden_size, num_bases, out_size, num_encoder_layers, num_decoder_layers):
        super(RGCNNodeRegression, self).__init__()
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

def get_model(g, in_size, out_size, hyperparameters, model_file=None, device="cpu"):
    model = RGCNNodeRegression(g.etypes,
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

def evaluate(model, features, labels, loss_fn, dataloader, target_ntype, device="cpu"):

    model.eval()
    predictions = []
    node_ids = []
    for i, (input_nodes, output_nodes, subgraphs) in enumerate(dataloader):
        batch_features = {ntype: {feat_name: feat[node_idx].to(device) for feat_name, feat in features[ntype].items()}
                          for ntype, node_idx in input_nodes.items()}
        subgraphs = [subgraph.to(device) for subgraph in subgraphs]
        logits = model(subgraphs, batch_features)
        predictions.append(logits)
        node_ids.append(output_nodes[target_ntype])

    predictions = torch.cat(predictions, dim=0).detach()
    labels = labels[torch.cat(node_ids, dim=0)].to(device)
    loss = loss_fn(predictions, labels).item()

    return loss

def train_n_epochs(model, optimizer, features, labels, loss_fn, train_dataloader, validation_dataloader, n_epochs,
                   target_ntype, device, model_path, model_file="model.pt", train_log_freq=5):

    best_eval_metric = np.Inf
    for epoch in range(n_epochs):
        t1 = time.time()
        for i, (input_nodes, output_nodes, subgraphs) in enumerate(train_dataloader):
            batch_features = {ntype: {feat_name: feat[node_idx].to(device) for feat_name, feat in features[ntype].items()}
                              for ntype, node_idx in input_nodes.items()}
            subgraphs = [subgraph.to(device) for subgraph in subgraphs]
            logits = model(subgraphs, batch_features)

            loss = loss_fn(logits, labels[output_nodes[target_ntype]].to(device))

            if (i+1)%train_log_freq == 0:
                print("Train Loss: {:.4f}".format(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch {:05d}:{:05d} | Epoch Time(s) {:.4f}".format(epoch + 1, n_epochs, time.time() - t1))

        val_loss = evaluate(model, features, labels, loss_fn, validation_dataloader, target_ntype, device=device)
        if val_loss < best_eval_metric:
            print("Validation loss: {:.4f}".format(val_loss))
            model.save(os.path.join(model_path, model_file))
            save_eval_metrics({"MSE": val_loss}, model_path)
            best_eval_metric = val_loss


def train(data_path, model_path, devices, hyperparameters):
    print("Training config: data_path: {}, model_path: {}, devices: {} hyperparameters: {}".format(data_path,
                                                                                                  model_path,
                                                                                                  devices,
                                                                                                  hyperparameters))
    device_type = get_device_type(devices)

    graphloader = GraphLoader(data_path)
    g = graphloader.graph

    print("Loaded graph: {}".format(g))

    target_ntype = hyperparameters["target_ntype"]

    train_nids = {target_ntype: get_train_nids(g, target_ntype)}
    sampler = dgl.dataloading.MultiLayerNeighborSampler([args.num_neighbors] * args.num_encoder_layers)
    train_dataloader = dgl.dataloading.NodeDataLoader(g, train_nids, sampler, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=0)

    valid_nids = {target_ntype: get_valid_nids(g, target_ntype)}
    val_sampler = dgl.dataloading.MultiLayerNeighborSampler([args.num_neighbors] * args.num_encoder_layers)
    val_dataloader = dgl.dataloading.NodeDataLoader(g, valid_nids, val_sampler, batch_size=args.batch_size, shuffle=False,
                                                num_workers=0)

    features_dict = graphloader.get_node_features()
    input_sizes = {ntype: {feat_name: features[feat_name].shape[1] for feat_name in features}
                   for ntype, features in features_dict.items()}
    print("Got input features with shape graph: {}".format(input_sizes))

    labels = get_node_labels(g, target_ntype).float()
    output_size = 1
    print("Got training labels with shape graph: {}".format(labels.shape))

    loss_fn = nn.MSELoss()

    model = get_model(g, input_sizes, output_size, hyperparameters, device=device_type)
    print("Created model: {}".format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"],
                                 weight_decay=hyperparameters["weight_decay"])

    print("Starting model training")

    train_n_epochs(model, optimizer, features_dict, labels, loss_fn, train_dataloader, val_dataloader,
                   hyperparameters["n_epochs"], hyperparameters["target_ntype"], device_type, model_path)


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

    target_ntype = hyperparameters["target_ntype"]

    features_dict = graphloader.get_node_features()
    input_sizes = {ntype: {feat_name: features[feat_name].shape[1] for feat_name in features}
                   for ntype, features in features_dict.items()}
    output_size = 1
    print("Got input features with shape graph: {}".format(input_sizes))

    model = get_model(g, input_sizes, output_size, hyperparameters, device=device_type, model_file=os.path.join(model_path, "model.pt"))
    print("Created model with saved parameters: {}".format(model))

    print("Getting model embeddings")
    node_embeddings = model.get_embeddings(g, features_dict, batch_size=hyperparameters['batch_size'], device=device_type, num_workers=0)
    predictions = torch.sigmoid(model.decoder(node_embeddings[target_ntype]))

    print("Saving model artifacts")
    save_node_prediction_model_artifacts(model_path, predictions, graphloader, hyperparameters, embeddings=node_embeddings)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action='store_true', default=False, help='Whether script is running locally')
    parser.add_argument("--name", type=str, default='rgcn-node-regression')
    parser.add_argument("--model", type=str, default='custom')
    parser.add_argument("--task", type=str, default='node_regression')
    parser.add_argument("--property", type=str, default='rating')
    parser.add_argument("--target_ntype", type=str, default='rating')
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
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
        data_path, model_path, devices = './data', './output', [-1]
    else:
        data_path, model_path, devices = get_training_config()

    train(data_path, model_path, devices, vars(args))