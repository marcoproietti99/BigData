from os import mkdir
from os.path import join, abspath, exists, dirname
from argparse import ArgumentParser





def load():
    parser = ArgumentParser()
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--per', default=80, type=int)
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_neurons', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--k', default=7, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--grid_search', default=True, type=bool)

    args = parser.parse_args()
    return args
