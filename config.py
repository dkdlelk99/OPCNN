import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = CN()
    cfg.device = 'cuda:0'
    cfg.seed = [0,1,2,3,4]
    cfg.runs = 5  # Number of runs with random init
    cfg.gnn = CN()
    cfg.lm = CN()
    cfg.resnet = CN()
    cfg.distill = CN()
    
    # ------------------------------------------------------------------------ #
    # Dataset options
    # ------------------------------------------------------------------------ #
    cfg.dataset.name = 'BBB' # 'BBB' or 'logP'
    cfg.dataset.range = 'all' # 'all' or 'approved'
    cfg.dataset.target_task = 'regression' # 'classification' or 'regression'
    cfg.dataset.split_method = 'scaffold' #['random', 'scaffold', 'random_scaffold']
    
    cfg.dataset.train_prop = 0.8
    cfg.dataset.val_prop = 0.1
    cfg.dataset.test_prop = 0.1

    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    cfg.gnn.model.name = 'gcn'
    cfg.gnn.model.num_layers = 5
    cfg.gnn.model.hidden_dim = 128  # <----------------------------------------------------------------------------
    cfg.gnn.model.max_nodes = 132  # bbbp: 132   bace: 97   clintonx: 136

    cfg.gnn.train = CN()
    cfg.gnn.train.weight_decay = 0.0
    cfg.gnn.train.epochs = 800
    # cfg.gnn.train.early_stop = 50
    cfg.gnn.train.lr = 0.005
    cfg.gnn.train.wd = 0.0005  # weight_decay
    cfg.gnn.train.dropout = 0.3
    cfg.gnn.train.batch_size = 10000000

    # ------------------------------------------------------------------------ #
    # ResNet Model options
    # ------------------------------------------------------------------------ #
    cfg.resnet.model = CN()
    cfg.resnet.model.name = 'resnet'
    cfg.resnet.model.vector_dim = 100
    cfg.resnet.model.channel = 32
    cfg.resnet.model.output_dim = 1
    
    # ------------------------------------------------------------------------ #
    # Distill Model options
    # ------------------------------------------------------------------------ #
    cfg.distill.model = CN()
    cfg.distill.model.name = 'mlp'
    cfg.distill.model.num_layers = 3
    cfg.distill.model.hidden_dim = 32 # <----------------------------------------------------------------------------
    cfg.distill.model.max_nodes = 132 # bbbp: 132   bace: 97   clintonx: 136

    cfg.distill.train = CN()
    cfg.distill.train.weight_decay = 0.0
    cfg.distill.train.epochs = 800
    cfg.distill.train.lr = 0.005
    cfg.distill.train.wd = 0.0005 # weight_decay
    cfg.distill.train.dropout = 0.3
    cfg.distill.train.batch_size = 100000000
    cfg.distill.train.alpha = 0.1
    cfg.distill.train.beta = 0.1


    return cfg



def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER, help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())


