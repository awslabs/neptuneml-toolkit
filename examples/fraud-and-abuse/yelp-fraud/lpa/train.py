import argparse
import os
import time
from functools import partial
from neptuneml_toolkit.train import get_training_config, get_train_nids, get_valid_nids, get_node_labels
from neptuneml_toolkit.metrics import classification_report, save_eval_metrics
from neptuneml_toolkit.transform import save_node_prediction_model_artifacts, normalize_hyperparameter_keys
from neptuneml_toolkit.utils import get_device_type
from neptuneml_toolkit.graphloader import GraphLoader
from neptuneml_toolkit.modelzoo import LabelPropagation, MLP
from sklearn.metrics import roc_auc_score

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelPropNodeClassification(nn.Module):
    def __init__(self, num_gs, in_size, hidden_size, out_size, num_lp_layers, num_mlp_layers, alpha, adj_norm):
        super(LabelPropNodeClassification, self).__init__()
        self.feature_mlp = MLP(in_size, out_size, hidden_units=hidden_size, num_layers=num_mlp_layers)
        self.label_prop = LabelPropagation(num_lp_layers, alpha, adj_norm=adj_norm)
        self.ensemble_weights = nn.Parameter(torch.Tensor(num_gs+1))
        nn.init.uniform_(self.ensemble_weights)

    def propagate(self, gs, y, mask=None):
        return [self.label_prop(g, y, mask=mask) for g in gs]

    def forward(self, h, device):
        return self.feature_mlp(h.to(device))

    def get_predictions(self, gs, h, labels, device):
        mlp_labels = [self.forward(h, device)]
        propagated_labels = [l.to(device) for l in self.propagate(gs, labels)]
        return torch.sum(torch.stack(mlp_labels + propagated_labels), dim=0)

    def save(self, model_file):
        torch.save({'model_state_dict': self.state_dict()}, model_file)

def get_model(num_subgraphs, in_size, out_size, hyperparameters, model_file=None, device="cpu"):
    model = LabelPropNodeClassification(num_subgraphs,
                                        in_size,
                                        int(hyperparameters['hidden_size']),
                                        out_size,
                                        int(hyperparameters['num_lp_layers']),
                                        int(hyperparameters['num_mlp_layers']),
                                        float(hyperparameters['alpha']),
                                        hyperparameters['adj_norm'])
    if model_file is not None:
        model_dict = torch.load(model_file, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['model_state_dict'])

    model = model.to(device)
    return model

def evaluate(model_predictions, labels, loss_fn, propagated_labels, valid_nids, device="cpu"):
    all_predictions = [F.softmax(model_predictions[valid_nids], dim=1).cpu()] + [prop_labels[valid_nids] for prop_labels in propagated_labels]
    predictions = torch.sum(torch.stack(all_predictions), dim=0)
    labels = labels[valid_nids].to(device)
    loss = loss_fn(model_predictions[valid_nids], labels).item()
    metric = roc_auc_score(labels.cpu().numpy(), predictions[:, 1].detach().cpu().numpy())
    report = classification_report(labels, predictions.argmax(dim=1))

    return metric, loss, report

def train_n_epochs(model, optimizer, features, labels, loss_fn, subgraphs, n_epochs, train_nids, valid_nids, device,
                   model_path, model_file="model.pt"):

    best_eval_metric = 0
    propagated_labels = model.propagate(subgraphs, labels, train_nids)
    for epoch in range(n_epochs):
        t1 = time.time()

        predictions = model(features, device)

        loss = loss_fn(predictions[train_nids], labels[train_nids].to(device))

        print("Train Loss: {:.4f}".format(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch {:05d}:{:05d} | Epoch Time(s) {:.4f}".format(epoch + 1, n_epochs, time.time() - t1))

        metric, val_loss, report = evaluate(predictions, labels, loss_fn, propagated_labels, valid_nids, device=device)
        if metric > best_eval_metric:
            print("Validation ROC AUC Score: {:.4f} | Validation loss: {:.4f}".format(metric, val_loss))
            model.save(os.path.join(model_path, model_file))
            report["roc_auc_score"] = metric
            save_eval_metrics(report, model_path)
            best_eval_metric = metric


def train(data_path, model_path, devices, hyperparameters):
    print("Training config: data_path: {}, model_path: {}, devices: {} hyperparameters: {}".format(data_path,
                                                                                                  model_path,
                                                                                                  devices,
                                                                                                  hyperparameters))
    device_type = get_device_type(devices)

    graphloader = GraphLoader(data_path)
    g = graphloader.graph

    print("Loaded graph: {}".format(g))

    target_ntype = hyperparameters['target_ntype']

    etypes = [etype for etype in g.canonical_etypes if 'rev-' not in etype[1]]
    reverse_etypes = [(dst, 'rev-' + etype, src) for (src, etype, dst) in etypes]

    subgraphs = [dgl.to_homogeneous(dgl.edge_type_subgraph(g, etypes=list(etype)))
                 for etype in zip(etypes, reverse_etypes)]

    features_dict = graphloader.get_node_features()[target_ntype]
    features = torch.cat(list(features_dict.values()), dim=1)
    input_size = features.shape[1]
    print("Got input features with shape graph: {}".format(features.shape))

    train_nids, valid_nids = get_train_nids(g, target_ntype), get_valid_nids(g, target_ntype)

    labels = get_node_labels(g, target_ntype)
    output_size = torch.max(labels).item() + 1
    print("Got training labels with shape: {}".format(labels.shape))
    num_labels = float(labels.shape[0])
    loss_scaler = torch.FloatTensor([labels.sum()/num_labels, 1 - (labels.sum()/num_labels)]).to(device_type)
    print("Scaling weights for loss: {}".format(loss_scaler))
    loss_fn = partial(F.cross_entropy, weight=loss_scaler)

    model = get_model(len(subgraphs), input_size, output_size, hyperparameters, device=device_type)
    print("Created model: {}".format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"],
                                 weight_decay=hyperparameters["weight_decay"])

    print("Starting model training")

    train_n_epochs(model, optimizer, features, labels, loss_fn, subgraphs, hyperparameters["n_epochs"],
                   train_nids, valid_nids, device_type, model_path)


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

    target_ntype = hyperparameters['target_ntype']

    etypes = [etype for etype in g.canonical_etypes if 'rev-' not in etype[1]]
    reverse_etypes = [(dst, 'rev-'+etype, src) for (src, etype, dst) in etypes]

    subgraphs = [dgl.to_homogeneous(dgl.edge_type_subgraph(g, etypes=list(etype)))
                 for etype in zip(etypes, reverse_etypes)]

    features_dict = graphloader.get_node_features()[target_ntype]
    features = torch.cat(list(features_dict.values()), dim=1)
    input_size = features.shape[1]
    print("Got input features with shape graph: {}".format(features.shape))

    labels = get_node_labels(g, target_ntype)

    model = get_model(len(subgraphs), input_size, 2, hyperparameters, device=device_type,
                      model_file=os.path.join(model_path, "model.pt"))
    print("Created model with saved parameters: {}".format(model))

    print("Getting model predictions")
    predictions = model.get_predictions(subgraphs, features, labels, device_type)

    print("Saving model artifacts")
    save_node_prediction_model_artifacts(model_path, predictions, graphloader, hyperparameters)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action='store_true', default=False, help='Whether script is running locally')
    parser.add_argument("--name", type=str, default='han-node-class')
    parser.add_argument("--model", type=str, default='custom')
    parser.add_argument("--task", type=str, default='node_class')
    parser.add_argument("--property", type=str, default='label')
    parser.add_argument("--target_ntype", type=str, default='review')
    parser.add_argument("--target-ntype", type=str, default="review", help='the target node type for classification')
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default = 128)
    parser.add_argument("--adj-norm", type=str, default='DAD', choices=['DAD', 'DA', 'AD'])
    parser.add_argument("--alpha", type=float, default=.75)
    parser.add_argument("--num-lp-layers", type=int, default=1)
    parser.add_argument("--num-mlp-layers", type=int, default=3)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.local:
        data_path, model_path, devices = './data', './output', [0]
    else:
        data_path, model_path, devices = get_training_config()

    train(data_path, model_path, devices, vars(args))