import argparse
import os
import time
from functools import partial
from neptuneml_toolkit.train import get_training_config, get_train_nids, get_valid_nids, get_node_labels
from neptuneml_toolkit.metrics import classification_report, save_eval_metrics
from neptuneml_toolkit.transform import save_node_prediction_model_artifacts, normalize_hyperparameter_keys
from neptuneml_toolkit.utils import get_device_type
from neptuneml_toolkit.graphloader import GraphLoader
from neptuneml_toolkit.modelzoo import HANEncoder, MLP
from neptuneml_toolkit.dataloading import MultiSubgraphNodeDataLoader, MetapathListSampler
from dgl import metapath_reachable_graph
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

class HANNodeClassification(nn.Module):
    def __init__(self, num_metapaths, in_size, hidden_size, num_heads, out_size, num_encoder_layers, num_decoder_layers):
        super(HANNodeClassification, self).__init__()
        self.encoder = HANEncoder(num_metapaths, in_size, hidden_size, num_encoder_layers, num_heads=num_heads)
        self.decoder = MLP(hidden_size, out_size, num_layers=num_decoder_layers)

    def forward(self, g, h):
        embeddings = self.encoder(g, h)
        return self.decoder(embeddings)

    def get_embeddings(self, g, h):
        return self.encoder(g, h)

    def save(self, model_file):
        torch.save({'model_state_dict': self.state_dict()}, model_file)

def get_model(in_size, out_size, hyperparameters, model_file=None, device="cpu"):
    model = HANNodeClassification(len(hyperparameters['metapaths']),
                                  in_size,
                                  int(hyperparameters['hidden_size']),
                                  int(hyperparameters['num_heads']),
                                  out_size,
                                  int(hyperparameters['num_encoder_layers']),
                                  int(hyperparameters['num_decoder_layers']))
    if model_file is not None:
        model_dict = torch.load(model_file, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['model_state_dict'])

    model = model.to(device)
    return model

def evaluate(model, features, labels, loss_fn, dataloader, device="cpu"):

    model.eval()
    predictions = []
    node_ids = []
    for i, (input_nodes, output_nodes, subgraphs) in enumerate(dataloader):
        batch_features = [features[node_idx].to(device) for node_idx in input_nodes]
        subgraphs = [[subgraph.to(device) for subgraph in block] for block in subgraphs]
        logits = model(subgraphs, batch_features)
        predictions.append(logits)
        node_ids.append(output_nodes[0])

    predictions = torch.cat(predictions, dim=0).detach()
    labels = labels[torch.cat(node_ids, dim=0)].to(device)
    loss = loss_fn(predictions, labels).item()
    metric = roc_auc_score(labels.cpu().numpy(), F.softmax(predictions, dim=1)[:, 1].cpu().numpy())
    report = classification_report(labels, predictions.argmax(dim=1))

    return metric, loss, report

def train_n_epochs(model, optimizer, features, labels, loss_fn, train_dataloader, validation_dataloader, n_epochs,
                   device, model_path, model_file="model.pt", train_log_freq=5):

    best_eval_metric = 0
    for epoch in range(n_epochs):
        t1 = time.time()
        for i, (input_nodes, output_nodes, subgraphs) in enumerate(train_dataloader):

            batch_features = [features[node_idx].to(device) for node_idx in input_nodes]
            subgraphs = [[subgraph.to(device) for subgraph in block] for block in subgraphs]
            logits = model(subgraphs, batch_features)

            loss = loss_fn(logits, labels[output_nodes[0]].to(device))

            if (i+1)%train_log_freq == 0:
                print("Train Loss: {:.4f}".format(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch {:05d}:{:05d} | Epoch Time(s) {:.4f}".format(epoch + 1, n_epochs, time.time() - t1))

        metric, val_loss, report = evaluate(model, features, labels, loss_fn, validation_dataloader, device=device)
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

    target_ntype = hyperparameters["target_ntype"]
    metapaths = [mp.split(";") for mp in hyperparameters["metapaths"].split(",")]
    hyperparameters["metapaths"] = metapaths
    print("Creating metapath sampler with metapath lists: {}".format(metapaths))
    num_neighbors = [hyperparameters["num_neighbors"]] * hyperparameters["num_encoder_layers"]
    metapath_sampler = MetapathListSampler(g, metapaths, num_neighbors=num_neighbors)


    train_nids = {target_ntype: get_train_nids(g, target_ntype)}
    train_dataloader = MultiSubgraphNodeDataLoader(g, train_nids, metapath_sampler,
                                                   batch_size=hyperparameters["batch_size"], shuffle=True)
    valid_nids = {target_ntype: get_valid_nids(g, target_ntype)}
    valid_dataloader = MultiSubgraphNodeDataLoader(g, valid_nids, metapath_sampler,
                                                   batch_size=hyperparameters["batch_size"], shuffle=False)

    features_dict = graphloader.get_node_features()[target_ntype]
    features = torch.cat(list(features_dict.values()), dim=1)
    input_size = features.shape[1]
    print("Got input features with shape: {}".format(features.shape))

    labels = get_node_labels(g, target_ntype)
    output_size = torch.max(labels).item() + 1
    print("Got training labels with shape: {}".format(labels.shape))
    num_labels = float(labels.shape[0])
    loss_scaler = torch.FloatTensor([labels.sum()/num_labels, 1 - (labels.sum()/num_labels)]).to(device_type)
    print("Scaling weights for loss: {}".format(loss_scaler))
    loss_fn = partial(F.cross_entropy, weight=loss_scaler)

    model = get_model(input_size, output_size, hyperparameters, device=device_type, )
    print("Created model: {}".format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"],
                                 weight_decay=hyperparameters["weight_decay"])

    print("Starting model training")

    train_n_epochs(model, optimizer, features, labels, loss_fn, train_dataloader, valid_dataloader,
                   hyperparameters["n_epochs"], device_type, model_path)


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
    metapaths = [mp.split(";") for mp in hyperparameters["metapaths"].split(",")]
    hyperparameters["metapaths"] = metapaths
    print("Creating metapath graphs with metapath lists: {}".format(metapaths))
    metapath_graphs = [metapath_reachable_graph(g, metapath) for metapath in metapaths]

    features_dict = graphloader.get_node_features()[target_ntype]
    features = torch.cat(list(features_dict.values()), dim=1)
    input_size = features.shape[1]
    print("Got input features with shape graph: {}".format(features.shape))

    model = get_model(input_size, 2, hyperparameters, device=device_type, model_file=os.path.join(model_path, "model.pt"))
    print("Created model with saved parameters: {}".format(model))

    print("Getting model embeddings")
    node_embeddings = model.get_embeddings(metapath_graphs, [features]*len(metapath_graphs))
    predictions = F.softmax(model.decoder(node_embeddings), dim=1)

    print("Saving model artifacts")
    save_node_prediction_model_artifacts(model_path, predictions, graphloader, hyperparameters, embeddings=node_embeddings)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action='store_true', default=False, help='Whether script is running locally')
    parser.add_argument("--name", type=str, default='han-node-class')
    parser.add_argument("--model", type=str, default='custom')
    parser.add_argument("--task", type=str, default='node_class')
    parser.add_argument("--property", type=str, default='label')
    parser.add_argument("--target_ntype", type=str, default='review')
    parser.add_argument("--metapaths", type=str, default="same_user,same_month,same_rating",
                        help='metapaths separated by "," where each metapath is the sequence of edges separated by ";"')
    parser.add_argument("--target-ntype", type=str, default="review", help='the target node type for classification')
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default = 128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-decoder-layers", type=int, default=1)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.local:
        data_path, model_path, devices = './data', './output', [0]
    else:
        data_path, model_path, devices = get_training_config()

    train(data_path, model_path, devices, vars(args))