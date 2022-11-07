import argparse
import os
import xgboost as xgb
import pickle as pkl
import numpy as np

from neptuneml_toolkit.train import get_training_config, get_train_nids, get_valid_nids, get_node_labels
from neptuneml_toolkit.metrics import save_eval_metrics
from neptuneml_toolkit.transform import save_node_prediction_model_artifacts, normalize_hyperparameter_keys
from neptuneml_toolkit.utils import get_device_type
from neptuneml_toolkit.graphloader import GraphLoader

def xgb_train(params, dtrain, evals, num_boost_round, model_dir):
    booster = xgb.train(params=params,
                        dtrain=dtrain,
                        evals=evals,
                        num_boost_round=num_boost_round)

    model_location = model_dir + '/xgboost-model'
    pkl.dump(booster, open(model_location, 'wb'))
    print("Stored trained model at {}".format(model_location))

def model_fn(model_dir):
    model_file = 'xgboost-model'
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster

def train(data_path, model_path, devices, hyperparameters):
    graphloader = GraphLoader(data_path)
    g = graphloader.graph
    print("Loaded graph: {}".format(g))
    target_ntype = hyperparameters["target_ntype"]
    train_nids, valid_nids= get_train_nids(g, target_ntype), get_valid_nids(g, target_ntype)

    features_dict = graphloader.get_node_features()[target_ntype]
    features = np.concatenate(list(features_dict.values()), axis=1)
    print("Got input features with shape: {}".format(features.shape))

    labels = get_node_labels(g, target_ntype)
    print("Got training labels with shape: {}".format(labels.shape))

    dtrain = xgb.DMatrix(data=features[train_nids], label=labels[train_nids].numpy())
    dval = xgb.DMatrix(data=features[valid_nids], label=labels[valid_nids].numpy())
    watchlist = [(dtrain, 'train'), (dval, 'validation')]

    train_hp = {
        'max_depth': int(hyperparameters['max_depth']),
        'eta': float(hyperparameters['eta']),
        'gamma': float(hyperparameters['gamma']),
        'min_child_weight': int(hyperparameters['min_child_weight']),
        'subsample': float(hyperparameters['subsample']),
        'verbosity': int(hyperparameters['verbosity']),
        'objective': hyperparameters['objective'],
        'eval_metric': hyperparameters['eval_metric'],
        'tree_method': 'auto',
        'predictor': 'auto',
    }

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=model_path)


    xgb_train(**xgb_train_args)

def transform(data_path, model_path, devices, hyperparameters):
    graphloader = GraphLoader(data_path)
    g = graphloader.graph
    print("Loaded graph: {}".format(g))
    target_ntype = hyperparameters["target_ntype"]

    features_dict = graphloader.get_node_features()[target_ntype]
    features = np.concatenate(list(features_dict.values()), axis=1)
    dmatrix = xgb.DMatrix(data=features)
    print("Got input features with shape: {}".format(features.shape))

    print("Getting model predictions")
    preb_proba = model_fn(model_path).predict(dmatrix)
    predictions = np.vstack((1-preb_proba, preb_proba)).T

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
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--min-child-weight", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.7)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--num-round", type=int, default=50)
    parser.add_argument("--eval-metric", type=str, default="auc")
    parser.add_argument("--objective", type=str, default='binary:logistic')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.local:
        data_path, model_path, devices = './tmp', './out', [0]
    else:
        data_path, model_path, devices = get_training_config()

    train(data_path, model_path, devices, vars(args))