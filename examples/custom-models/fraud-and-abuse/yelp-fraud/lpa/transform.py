import argparse
from neptuneml_toolkit.transform import get_transform_config
from train import transform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action='store_true', default=False, help='Whether script is running locally')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.local:
        hyperparameters = {'target_ntype': 'review', 'task': 'node_class', 'model':'lpa', 'name': 'lpa-node-class', 'property': 'label',
                           'batch-size': 1024, 'lr': 0.01, 'weight-decay': 0.0, 'n_epochs': 2, 'alpha': .75,
                           'hidden-size': 128, 'adj-norm': 'DAD', 'num-lp-layers': 1, 'num-mlp-layers': 3}
        data_path, model_path, devices = './data', './output', [-1]
    else:
        data_path, model_path, devices, hyperparameters = get_transform_config()

    transform(data_path, model_path, devices, hyperparameters)