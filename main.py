import argparse
import os
import os.path as osp
from shutil import copy2

import torch
from torch import optim

import hdbayesopt
import mesh2graph
import net
import train
import utils
from torch_geometric.data import DataLoader
from torch_geometric.data import HeartGraphDataset
from torch_geometric.data import MySimpleHeartDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    """
    Args:
        config: json file with hyperparams and exp settings
        seed: random seed value
        stage: 1 for traing VAE, 2 for optimization,  and 12 for both
        logging: 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='params', help='config filename')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--logging', type=bool, default=True, help='logging')
    parser.add_argument('--stage', type=int, default=1, help='1.VAE, 2.BO, 12.VAE_BO, 3.Eval VAE')

    args = parser.parse_args()
    return args



def learn_vae(hparams, training=True):
    """Generative modeling of the HD tissue properties
    """
    vae_type = hparams.model_type
    batch_size = hparams.batch_size
    num_epochs = hparams.num_epochs
    num_meshfree = hparams.num_meshfree

    # directory path for training and testing datasets
    data_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                        'data', 'training', hparams.heart_name)

    # directory path to save the model/results
    model_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                         'experiments', vae_type, hparams.model_name)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    copy2(json_path, model_dir)

    # logging the training procedure
    # if args.logging:
    #    sys.stdout = open(model_dir+'/log.txt','wt')

    if vae_type == 'gvae':  # for gvae
        graph_dir = osp.join(data_dir, 'raw', hparams.graph_name)
        # Create graph and load graph information
        if training and hparams.makegraph:
            g = mesh2graph.GraphPyramid(hparams.heart_name, num_meshfree)
            g.make_graph()
        graphparams = net.get_graphparams(graph_dir, device, batch_size)

        # initialize datasets and dataloader
        train_dataset = HeartGraphDataset(root=data_dir, num_meshfree=num_meshfree,
                                          mesh_graph=graphparams["g"], train=True)
        test_dataset = HeartGraphDataset(root=data_dir, num_meshfree=num_meshfree,
                                         mesh_graph=graphparams["g"], train=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # initialize the model
        model = net.GraphVAE(hparams)
        model.set_graphs(graphparams)
        model.to(device)

    else:  # for fvae
        # initialize datasets and dataloader
        train_dataset = MySimpleHeartDataset(data_dir, num_meshfree, train=True)
        test_dataset = MySimpleHeartDataset(data_dir, num_meshfree, train=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=True)

        # initialize the model
        model = net.FcVAE(hparams).to(device)

    corMfree = train_dataset.getCorMfree()
    if training:
        optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
        train.train_vae(model, optimizer, train_loader, test_loader,
                        model_dir, num_epochs, batch_size, corMfree)
    else:
        model.load_state_dict(torch.load(model_dir + '/' +hparams.vae_latest))  
        model = model.eval().to(device)
        train.eval_vae(model, train_loader, test_loader, model_dir, batch_size, corMfree)

        


def optimize_params(hparams):
    """HD Bayesian optimization via an embedded generative model
    """
    # read paths
    model_name = hparams.model_name  # vae modelname
    vae_type = hparams.model_type  # fvae vs gvae
    batch_size = hparams.batch_size  #
    num_epochs = hparams.num_epochs  #
    num_meshfree = hparams.num_meshfree  #

    p_dim = num_meshfree  # number of meshfree nodes
    files = tuple(hparams.opti_exps)  # files to optimize
    acq_list = tuple(hparams.acq_funs)  # read the acquisation functions)

    # read the trained vae model
    #device = 'cpu'
    # directory path to save the model/results
    model_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                         'experiments', vae_type, model_name)
    if osp.exists(model_dir):
        if vae_type == 'fvae':
            vae = net.FcVAE(hparams)
        else:
            # directory path for training and testing datasets
            data_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                                'data', 'training', hparams.heart_name)
            graph_dir = osp.join(data_dir, 'raw', hparams.graph_name)
            graphparams = net.get_graphparams(graph_dir, device, batch_size)
            vae = net.GraphVAE(hparams)
            vae.set_graphs(graphparams)

        
        vae.load_state_dict(torch.load(model_dir + '/' +hparams.vae_latest))  # Choose whatever GPU device number you want     
        vae = vae.eval()
        vae.to(device)

        z_mu_1 = None
        z_var_1 = None
        if 'ei_post_agg' in hparams.acq_funs:
            with open(model_dir + "/z_posterior.pkl", 'rb') as input:
                z_mu_1 = pickle.load(input)
                z_var_1 = pickle.load(input)

    else:
        # TODO
        print('This VAE has not been trained yet!! END program here!')
        return

    # path for the inputs and  outputs of optimization
    dpath = osp.join(osp.dirname(osp.realpath('__file__')),
                     'data', 'optimization', hparams.heart_name)
    rpath = osp.join(model_dir, hparams.heart_name)
    if not osp.exists(rpath):
        os.makedirs(rpath)

    # High dimensional BO
    hdbayesopt.hdbayesopt(vae, dpath, rpath, files, p_dim, acq_list,
                          niter=hparams.niter, inipts=hparams.inipts, 
                          z_mu_1=z_mu_1, z_var_1=z_var_1, verbose=hparams.verbose_bo)


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # filename of the params
    fname_config = args.config + '.json'
    # read the params file
    json_path = osp.join(osp.dirname(osp.realpath('__file__')), "config", fname_config)
    hparams = utils.Params(json_path)

    if args.stage == 1:  # generative modeling
        print('Stage 1: begin training vae ...')
        learn_vae(hparams)
        print('Training vae completed!')
        print('--------------------------------------')
    elif args.stage == 2:  # optimization
        print('Stage 2: begin optimization ...')
        optimize_params(hparams)
        print('Optimization completed!')
        print('--------------------------------------')
    elif args.stage == 12:
        print('Stage 1: begin training vae ...')
        learn_vae(hparams)  # Both generative modeling and optimization
        print('Training vae completed!')
        print('--------------------------------------')
        print('Stage 2: begin optimization ...')
        optimize_params(hparams)        
        print('Optimization completed!')
        print('--------------------------------------')
    elif args.stage == 3:
        print('Stage 3: begin evaluating vae ...')
        learn_vae(hparams, training=False) # Evaluate a vae
        print('Evaluating vae completed!')
        print('--------------------------------------')
    else:
        print('invalid stage option; valid 1, 2, 3 or 12')
