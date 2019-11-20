import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import DataLoader
from torch_geometric.data import HeartEmptyGraphDataset
from torch_geometric.nn import SplineConv


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FcVAE(nn.Module):
    """ VAE with fully connected layers on both encoders and decoders

    Args:
        hparams.batchsize: batch_size
        hparams.num_meshfree: number of meshfree nodes (units in the first layer)
        latent_dim: dimension of latent space
    """

    def __init__(self, hparams):
        super(FcVAE, self).__init__()
        self.batch_size = hparams.batch_size
        self.num_meshfree = hparams.num_meshfree
        self.latent_dim = hparams.latent_dim

        # probablistic encoder (or recognition) network
        self.fce1 = nn.Linear(hparams.num_meshfree, 512)
        self.fce2 = nn.Linear(512, 512)
        # self.fce2e = nn.Linear(512, 512)
        # self.fce2ee = nn.Linear(512, 512)
        self.fce31 = nn.Linear(512, hparams.latent_dim)
        self.fce32 = nn.Linear(512, hparams.latent_dim)

        # probablistic decoder (or generator) network
        self.fcd3 = nn.Linear(hparams.latent_dim, 512)
        # self.fcd2e = nn.Linear(512, 512)
        # self.fcd2ee = nn.Linear(512, 512)
        self.fcd2 = nn.Linear(512, 512)
        self.fcd11 = nn.Linear(512, hparams.num_meshfree)

    def encode(self, data):
        """encoder (recognition) network

        Args:
            data.x: input tissue properties in vector format
        """
        x = data.x
        # input is batch_size X num_nodes X num_features
        # reshape the input: batch_size X num_nodes
        x = x.view(-1, self.num_meshfree)
        # layer 1:  batch_size X num_nodes
        x = F.elu(self.fce1(x))
        # layer 2:  batch_size X num_nodes
        x = F.elu(self.fce2(x))
        # x = F.elu(self.fce2e(x))
        # x = F.elu(self.fce2ee(x))
        # layer z (z_mean, z_logvariance):  batch_size X num_representaitons
        mu = self.fce31(x)
        logvar = self.fce32(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization (draw a random sample from the p(z|x))

        Args:
            mu: mean from the probablistic encoder
            logvar: log variance from the probablistic decoder
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """ decoder (generator network)

        Args:
            z: a sample from the latent distribution
        """

        # layer 1: batch_size x num_representaions
        x = F.elu(self.fcd3(z))
        # layer 2: batch_size x num_representaions
        # x = F.elu(self.fcd2ee(x))
        # x = F.elu(self.fcd2e(x))
        x = F.elu(self.fcd2(x))
        # expectation of the generator
        u = self.fcd11(x)
        # logs = F.softplus(self.fcd12(x))
        return u

    def forward(self, x):
        """encoder - reparaemterization - decoder

        Output:
            u: decoded (reconstructed) signal
            mu:
            logvar:
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        u = self.decode(z)
        return u, mu, logvar


class GraphVAE(torch.nn.Module):
    """VAE with graph convolutional layers and pooling layers

    Args:
        hparams.batch_size:
        hparams.latent_dim:
        hparams.nf: number of features in each layer

    """

    def __init__(self, hparams):
        super(GraphVAE, self).__init__()

        self.nf = hparams.nf  # [1,8,16,32,64,128]
        self.batch_size = hparams.batch_size
        self.f_dim = self.nf[-1] * 1
        self.latent_dim = hparams.latent_dim

        # probablistic encoder (or recognition) network
        self.conv1 = SplineConv(self.nf[0], self.nf[1], dim=3, kernel_size=5, norm=False)
        self.conv2 = SplineConv(self.nf[1], self.nf[2], dim=3, kernel_size=5, norm=False)
        self.conv3 = SplineConv(self.nf[2], self.nf[3], dim=3, kernel_size=5, norm=False)
        self.conv4 = SplineConv(self.nf[3], self.nf[4], dim=3, kernel_size=5, norm=False)
        self.conv5 = SplineConv(self.nf[4], self.nf[5], dim=3, kernel_size=5, norm=False)

        self.fce1 = torch.nn.Conv2d(self.nf[-1], self.nf[-1], 1)
        self.fce21 = torch.nn.Conv2d(self.nf[-1], self.latent_dim, 1)
        self.fce22 = torch.nn.Conv2d(self.nf[-1], self.latent_dim, 1)

        # probablistic conv-decoder (or generator) network
        self.fcd3 = torch.nn.Conv2d(self.latent_dim, self.nf[-1], 1)
        self.fcd4 = torch.nn.Conv2d(self.nf[-1], self.nf[-1], 1)

        self.deconv5 = SplineConv(self.nf[5], self.nf[4], dim=3, kernel_size=5, norm=False)
        self.deconv4 = SplineConv(self.nf[4], self.nf[3], dim=3, kernel_size=5, norm=False)
        self.deconv3 = SplineConv(self.nf[3], self.nf[2], dim=3, kernel_size=5, norm=False)
        self.deconv2 = SplineConv(self.nf[2], self.nf[1], dim=3, kernel_size=5, norm=False)
        self.deconv1 = SplineConv(self.nf[1], self.nf[0], dim=3, kernel_size=5, norm=False)

    def set_graphs(self, gParams):
        """ Initialize the graph structure of the given mesh

        Args:
            gParams[bgi]: graph structure at ith level
            gParams[Pij]: matrix for downpooling
            gParams[P1n]: matrix for

        """
        self.bg = gParams["bg"]
        self.bg1 = gParams["bg1"]
        self.bg2 = gParams["bg2"]
        self.bg3 = gParams["bg3"]
        self.bg4 = gParams["bg4"]
        # self.bg5 = gParams["bg5"]
        # self.bg6 = gParams["bg6"]
        self.P01 = gParams["P01"]
        self.P12 = gParams["P12"]
        self.P23 = gParams["P23"]
        self.P34 = gParams["P34"]
        # self.P45 = gParams["P45"]
        # self.P56 = gParams["P56"]
        self.P10 = gParams["P10"]
        self.P21 = gParams["P21"]
        self.P32 = gParams["P32"]
        self.P43 = gParams["P43"]
        # self.P54 = gParams["P54"]
        # self.P65 = gParams["P65"]
        self.P1n = gParams["P1n"]
        self.Pn1 = gParams["Pn1"]

    def encode(self, data):
        """ graph convolutional encoder
        """

        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.x, data.edge_index, data.edge_attr  # (1230*bs) X f[0]
        x = F.elu(self.conv1(x, edge_index, edge_attr))  # (1230*bs) X f[1]
        x = x.view(self.batch_size, -1, self.nf[1])  # bs X 1230 X f[1]
        x = torch.matmul(self.P01, x)  # bs X 648 X f[1]
        # layer 2
        x, edge_index, edge_attr = \
            x.view(-1, self.nf[1]), self.bg1.edge_index, self.bg1.edge_attr
        x = F.elu(self.conv2(x, edge_index, edge_attr))  # 648*bs X f[2]
        x = x.view(self.batch_size, -1, self.nf[2])  # bs X 648 X f[2]
        x = torch.matmul(self.P12, x)  # bs X 347 X f[2]
        # layer 3
        x, edge_index, edge_attr = \
            x.view(-1, self.nf[2]), self.bg2.edge_index, self.bg2.edge_attr
        x = F.elu(self.conv3(x, edge_index, edge_attr))  # 347*bs X f[3]
        x = x.view(self.batch_size, -1, self.nf[3])  # bs X 347 X f[3]
        x = torch.matmul(self.P23, x)  # bs X 184 X f[3]
        # layer 4
        x, edge_index, edge_attr = \
            x.view(-1, self.nf[3]), self.bg3.edge_index, self.bg3.edge_attr
        x = F.elu(self.conv4(x, edge_index, edge_attr))  # 184*bs X f[4]
        x = x.view(self.batch_size, -1, self.nf[4])  # bs X 184 X f[4]
        x = torch.matmul(self.P34, x)  # bs X 97 X f[4]
        # layer 5
        x, edge_index, edge_attr = \
            x.view(-1, self.nf[4]), self.bg4.edge_index, self.bg4.edge_attr
        x = F.elu(self.conv5(x, edge_index, edge_attr))  # 184*bs X f[5]
        x = x.view(self.batch_size, -1, self.nf[5])  # bs X 97 X f[5]
        x = torch.matmul(self.Pn1, x)  # bs X 1 X f[5]
        # layer 6
        x = x.view(self.batch_size, self.nf[5], 1, 1)  # bs X 1 X f[5]
        x = F.elu(self.fce1(x))
        mu = self.fce21(x)
        logvar = self.fce22(x)
        return mu.view(self.batch_size, -1), logvar.view(self.batch_size, -1)

    def reparameterize(self, mu, logvar):
        """ reparameterization; draw a random sample from the p(z|x)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """ graph  convolutional decoder
        """

        z = z.view(self.batch_size, self.latent_dim, 1, 1)
        x = F.elu(self.fcd3(z))
        x = F.elu(self.fcd4(x))

        x = x.view(self.batch_size, -1, self.nf[5])  # bs X 1 X f[5]
        x = torch.matmul(self.P1n, x)  # bs X 95 X f[5]
        x, edge_index, edge_attr = \
            x.view(-1, self.nf[5]), self.bg4.edge_index, self.bg4.edge_attr
        x = F.elu(self.deconv5(x, edge_index, edge_attr))  # (bs*97) X f[4]

        x = x.view(self.batch_size, -1, self.nf[4])  # bs X 97 X f[4]
        x = torch.matmul(self.P43, x)  # bs X 184 X f[4]
        x, edge_index, edge_attr = \
            x.view(-1, self.nf[4]), self.bg3.edge_index, self.bg3.edge_attr
        x = F.elu(self.deconv4(x, edge_index, edge_attr))  # (bs*184) X f[3]

        x = x.view(self.batch_size, -1, self.nf[3])  # bs X 187 X f[3]
        x = torch.matmul(self.P32, x)  # bs X 351 X f[3]
        x, edge_index, edge_attr = \
            x.view(-1, self.nf[3]), self.bg2.edge_index, self.bg2.edge_attr
        x = F.elu(self.deconv3(x, edge_index, edge_attr))  # (bs*351) X f[2]

        x = x.view(self.batch_size, -1, self.nf[2])  # bs X 351 X f[2]
        x = torch.matmul(self.P21, x)  # bs X 646 X f[2]
        x, edge_index, edge_attr = \
            x.view(-1, self.nf[2]), self.bg1.edge_index, self.bg1.edge_attr
        x = F.elu(self.deconv2(x, edge_index, edge_attr))  # (bs*646) X f[1]

        x = x.view(self.batch_size, -1, self.nf[1])  # bs X 646 X f[1]
        x = torch.matmul(self.P10, x)  # bs X 1230 X f[1]
        x, edge_index, edge_attr = \
            x.view(-1, self.nf[1]), self.bg.edge_index, self.bg.edge_attr
        x = F.elu(self.deconv1(x, edge_index, edge_attr))  # (bs*1230) X f[0]

        x = x.view(self.batch_size, -1)
        return x

    def forward(self, data):
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    """ VAE Loss: Reconstruction + KL divergence losses summed over all elements and batch
    """
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def load_graph(filename):
    with open(filename + '.pickle', 'rb') as f:
        g = pickle.load(f)
        g1 = pickle.load(f)
        g2 = pickle.load(f)
        g3 = pickle.load(f)
        g4 = pickle.load(f)
        g5 = pickle.load(f)
        g6 = pickle.load(f)

        P10 = pickle.load(f)
        P21 = pickle.load(f)
        P32 = pickle.load(f)
        P43 = pickle.load(f)
        P54 = pickle.load(f)
        P65 = pickle.load(f)

    P01 = P10 / P10.sum(axis=0)
    P12 = P21 / P21.sum(axis=0)
    P23 = P32 / P32.sum(axis=0)
    P34 = P43 / P43.sum(axis=0)
    P45 = P54 / P54.sum(axis=0)
    P56 = P65 / P65.sum(axis=0)

    P01 = torch.from_numpy(np.transpose(P01)).float()
    P12 = torch.from_numpy(np.transpose(P12)).float()
    P23 = torch.from_numpy(np.transpose(P23)).float()
    P34 = torch.from_numpy(np.transpose(P34)).float()
    P45 = torch.from_numpy(np.transpose(P45)).float()
    P56 = torch.from_numpy(np.transpose(P56)).float()

    P10 = torch.from_numpy(P10).float()
    P21 = torch.from_numpy(P21).float()
    P32 = torch.from_numpy(P32).float()
    P43 = torch.from_numpy(P43).float()
    P54 = torch.from_numpy(P54).float()
    P65 = torch.from_numpy(P65).float()

    return g, g1, g2, g3, g4, g5, g6, P10, P21, P32, P43, P54, P65, P01, P12, P23, P34, P45, P56


def get_graphparams(filename, device, batch_size):
    g, g1, g2, g3, g4, g5, g6, P10, P21, P32, P43, P54, P65, P01, P12, P23, P34, P45, P56 = \
        load_graph(filename)

    num_nodes = [g.pos.shape[0], g1.pos.shape[0], g2.pos.shape[0], g3.pos.shape[0],
                 g4.pos.shape[0], g5.pos.shape[0], g6.pos.shape[0]]
    print(g)
    print(g1)
    print(g2)
    print(g3)
    print(g4)
    print(g5)
    print(g6)
    print('P21 requires_grad:', P21.requires_grad)
    print('number of nodes:', num_nodes)

    g_dataset = HeartEmptyGraphDataset(mesh_graph=g)
    g_loader = DataLoader(g_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg = next(iter(g_loader))

    g1_dataset = HeartEmptyGraphDataset(mesh_graph=g1)
    g1_loader = DataLoader(g1_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg1 = next(iter(g1_loader))

    g2_dataset = HeartEmptyGraphDataset(mesh_graph=g2)
    g2_loader = DataLoader(g2_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg2 = next(iter(g2_loader))

    g3_dataset = HeartEmptyGraphDataset(mesh_graph=g3)
    g3_loader = DataLoader(g3_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg3 = next(iter(g3_loader))

    g4_dataset = HeartEmptyGraphDataset(mesh_graph=g4)
    g4_loader = DataLoader(g4_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg4 = next(iter(g4_loader))

    g5_dataset = HeartEmptyGraphDataset(mesh_graph=g5)
    g5_loader = DataLoader(g5_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg5 = next(iter(g5_loader))

    g6_dataset = HeartEmptyGraphDataset(mesh_graph=g6)
    g6_loader = DataLoader(g6_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg6 = next(iter(g6_loader))

    P01 = P01.to(device)
    P12 = P12.to(device)
    P23 = P23.to(device)
    P10 = P10.to(device)
    P21 = P21.to(device)
    P32 = P32.to(device)

    P34 = P34.to(device)
    P45 = P45.to(device)
    P56 = P56.to(device)
    P43 = P43.to(device)
    P54 = P54.to(device)
    P65 = P65.to(device)

    bg1 = bg1.to(device)
    bg2 = bg2.to(device)
    bg3 = bg3.to(device)
    bg4 = bg4.to(device)
    bg5 = bg5.to(device)
    bg6 = bg6.to(device)
    bg = bg.to(device)

    P1n = np.ones((num_nodes[4], 1))
    Pn1 = P1n / P1n.sum(axis=0)
    Pn1 = torch.from_numpy(np.transpose(Pn1)).float()
    P1n = torch.from_numpy(P1n).float()
    P1n = P1n.to(device)
    Pn1 = Pn1.to(device)

    graphparams = {"bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4, "bg5": bg5, "bg6": bg6,
                   "P01": P01, "P12": P12, "P23": P23, "P34": P34, "P45": P45, "P56": P56,
                   "P10": P10, "P21": P21, "P32": P32, "P43": P43, "P54": P54, "P65": P65,
                   "P1n": P1n, "Pn1": Pn1, "num_nodes": num_nodes, "g": g, "bg": bg}
    # print(graphparams["g"])
    return graphparams
