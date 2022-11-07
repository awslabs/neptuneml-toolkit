import argparse
from neptuneml.transform import get_transform_config
from train import transform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action='store_true', default=False, help='Whether script is running locally')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.local:
        hyperparameters = {'metapaths': 'same_user,same_month,same_rating', 'target_ntype': 'review',
                           'task': 'node_class', 'model':'han', 'name': 'han-node-class', 'property': 'label',
                           'num-neighbors': 30, 'batch-size': 1024, 'lr': 0.01, 'weight-decay': 0.0, 'n_epochs': 2,
                           'hidden-size': 128, 'num-heads': 2, 'num-encoder-layers': 1, 'num-decoder-layers': 1}
        data_path, model_path, devices = './tmp', './out', [-1]
    else:
        data_path, model_path, devices, hyperparameters = get_transform_config()

    transform(data_path, model_path, devices, hyperparameters)