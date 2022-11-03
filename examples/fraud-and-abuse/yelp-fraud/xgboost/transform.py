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
        hyperparameters = {'predictor': 'auto', 'task': 'node_class', 'model':'xgboost', 'name': 'xgboost-node-class',
                           'property': 'label', 'target_ntype': 'review'}
        data_path, model_path, devices = './tmp', './out', [-1]
    else:
        data_path, model_path, devices, hyperparameters = get_transform_config()

    transform(data_path, model_path, devices, hyperparameters)