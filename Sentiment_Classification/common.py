import torch.nn as nn
import argparse


def weight_he_init(layers):
    init_fn = nn.init.kaiming_uniform
    for layer in layers:
        if isinstance(layer, nn.Linear):
            init_fn(layer.weight)


def weight_xavier_init(layers):
    init_fn = nn.init.xavier_uniform
    for layer in layers:
        if isinstance(layer, nn.Linear):
            init_fn(layer.weight)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
