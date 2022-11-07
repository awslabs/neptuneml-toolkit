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
        hyperparameters = {'target_ntype': 'rating', 'num-neighbors': 30, 'batch-size': 1024, 'lr': 0.01,
                           'task': 'node_regression', 'model': 'rgcn', 'name': 'rgcn-node-regression', 'property': 'rating',
                           'weight-decay': 0.0, 'n-epochs': 2, 'hidden-size': 128, 'num-bases': 2,
                           'num-encoder-layers': 2, 'num-decoder-layers': 1}
        data_path, model_path, devices = './data', './output', [-1]
    else:
        data_path, model_path, devices, hyperparameters = get_transform_config()

    transform(data_path, model_path, devices, hyperparameters)