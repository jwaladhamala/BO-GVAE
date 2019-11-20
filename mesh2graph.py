import copy
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import HeartNetSubset
from torch_geometric.nn.pool import *
from torch_geometric.utils import normalized_cut


class GraphPyramid():
    """Construct a graph for a given heart along with a graph hierarchy.
    For graph construction: Nodes are converted to vertices, edges are added between every node
    and it K nearest neighbor (criteria can be modified) and edge attributes between any two vertices
    is the normalized differences of Cartesian coordinates if an edge exists between the nodes
    , i.e., normalized [x1-x2, y1-y2, z1-z2] and 0 otherwise.
    
    For graph hierarchy, graph clustering method is used.
    
    Args:
        heart: name of the cardiac anatomy on which to construct the  graph and its hierarchy
        K: K in KNN for defining edge connectivity in the graph
    """

    def __init__(self, heart='case3', mfree=1230, K=6):
        """
        """
        self.path_in = osp.join(osp.dirname(osp.realpath('__file__')), 'data', 'training', heart)
        self.pre_transform = T.NNGraph(k=K)
        self.transform = T.Cartesian(cat=False)
        self.filename = osp.join(self.path_in, 'raw', heart)
        self.mfree = mfree

    def normalized_cut_2d(self, edge_index, pos):
        """ calculate the normalized cut 2d 
        """
        row, col = edge_index
        edge_attr = torch.norm(pos[row] - pos[col], dim=1)
        return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

    def save_graph(self, g, g1, g2, g3, g4, g5, g6, P10, P21, P32, P43, P54, P65):
        """save the graphs and the pooling matrices in a file
        """
        with open(self.filename + '_graclus_hier' + '.pickle', 'wb') as f:
            pickle.dump(g, f)
            pickle.dump(g1, f)
            pickle.dump(g2, f)
            pickle.dump(g3, f)
            pickle.dump(g4, f)
            pickle.dump(g5, f)
            pickle.dump(g6, f)

            pickle.dump(P10, f)
            pickle.dump(P21, f)
            pickle.dump(P32, f)
            pickle.dump(P43, f)
            pickle.dump(P54, f)
            pickle.dump(P65, f)

    def load_graph(self):
        """load the graphs and pooling matrices; used to test existing files
        """
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
        return g, g1, g2, g3, g4, g5, g6, P10, P21, P32, P01, P12, P23

    def clus_heart(self, d, method='graclus'):
        """Use graph clustering method to make a hierarchy of coarser-finer graphs
        
        Args:
            method: graph clustering method to use (options: graclus or voxel)
            d: a instance of Data class (a graph object)
        
        Output:
            P: transformation matrix from coarser to finer scale
            d_coarser: graph for the coarser scale
        """
        # clustering
        if (method == 'graclus'):
            weight = self.normalized_cut_2d(d.edge_index, d.pos)
            cluster = graclus(d.edge_index, weight, d.x.size(0))
        elif (method == 'voxel'):
            cluster = self.voxel_grid(d.pos, torch.tensor(np.zeros(d.pos.shape[0])), size=10)
        else:
            print('this clustering method has not been implemented')

        # get clusters assignments with consequitive numbers
        cluster, perm = self.consecutive_cluster(cluster)
        unique_cluster = np.unique(cluster)
        n, m = cluster.shape[0], unique_cluster.shape[0]  # num nodes, num clusters

        # transformaiton matrix that consists of num_nodes X num_clusters
        P = np.zeros((n, m))
        # P_{ij} = 1 if ith node in the original cluster was merged to jth node in coarser scale
        for j in range(m):
            i = np.where(cluster == int(unique_cluster[j]))
            P[i, j] = 1
        Pn = P / P.sum(axis=0)  # column normalize P
        PnT = torch.from_numpy(np.transpose(Pn)).float()  # PnT tranpose

        # the coarser scale features =  Pn^T*features
        # this is done for verification purpose only
        x = torch.mm(PnT, d.x)  # downsampled features
        pos = torch.mm(PnT, d.pos)  # downsampled coordinates (vertices)

        # convert into a new object of data class (graphical format)
        d_coarser = Data(x=x, pos=pos, y=d.y)
        d_coarser = self.pre_transform(d_coarser)
        d_coarser = self.transform(d_coarser)
        return P, d_coarser

    def declus_heart(self, gn_coarse, gn, Pr):
        """ Test the up-pooling matrix. Obtain finer scale features by patching operation.
        
        Args:
            gn: finer scale graph
            gn_coarse: coarser scale graph
            Pr: gn.features = Pr*gn_coarse.features  (obtain finer scale features)
        """
        x = torch.mm(torch.from_numpy(Pr).float(), gn_coarse.x)
        pos = gn.pos
        edge_index = gn.edge_index
        edge_attr = gn.edge_attr
        d_finer = Data(edge_attr=edge_attr, edge_index=edge_index,
                       x=x, pos=pos, y=gn_coarse.y)
        return d_finer

    def consecutive_cluster(self, src):
        """
        Args:
            src: cluster
        """
        unique, inv = torch.unique(src, sorted=True, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
        return inv, perm

    def scatter_plots(self, data, name='_', colorby=0):
        """visualize and save the graph data

        """
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        # x, y, z coordinates
        x = data.pos[:, 0]
        y = data.pos[:, 1]
        z = data.pos[:, 2]

        # color code in the figure
        if (colorby == 0):  # by features on the nodes
            features = data.x[:, 0]
            im = ax.scatter(x, y, z, s=30, c=features, cmap=plt.get_cmap('jet'), vmin=0.07, vmax=0.55)
        else:  # by labels on the nodes
            label = data.y
            if (len(label) > 1):
                im = ax.scatter(x, y, z, s=30, c=label, cmap=plt.get_cmap('jet'))
            else:
                im = ax.scatter(x, y, z, cmap=plt.get_cmap('jet'))
        plt.axis('off')
        fig.tight_layout()
        fig.savefig(self.filename + name + '_' + str(len(x)) + '.png', dpi=600,
                    bbox_inches='tight', transparent=True)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

    def make_graph(self, K=6):
        """Main function for constructing the graph and its hierarchy
        """

        # Create a graph on a subset of datapoints with pre-transform and transform properties 
        train_dataset = HeartNetSubset(self.path_in, True, pre_transform=self.pre_transform,
                                       transform=self.transform, mfree=self.mfree)
        # one instance of the graph class
        testdata = train_dataset[65]
        # scatter_plots(testdata) # plot a graph
        # print(testdata.x.shape)

        # begin creating a graph hierarchy (downpooling operation)
        g = copy.deepcopy(testdata)  # graph at the meshfree nodes level
        self.scatter_plots(g, name='pool')  # plot the graph
        P1, g1 = copy.deepcopy(self.clus_heart(g))
        self.scatter_plots(g1, name='pool')
        P2, g2 = copy.deepcopy(self.clus_heart(g1))
        self.scatter_plots(g2, name='pool')
        P3, g3 = copy.deepcopy(self.clus_heart(g2))
        self.scatter_plots(g3, name='pool')
        P4, g4 = copy.deepcopy(self.clus_heart(g3))
        self.scatter_plots(g4, name='pool')
        P5, g5 = copy.deepcopy(self.clus_heart(g4))
        self.scatter_plots(g5, name='pool')
        P6, g6 = copy.deepcopy(self.clus_heart(g5))
        self.scatter_plots(g6, name='pool')

        # uppooling operation (for visualization purpose)
        g5_d = copy.deepcopy(self.declus_heart(g6, g5, P6))
        self.scatter_plots(g5_d, name='unpool')
        g4_d = copy.deepcopy(self.declus_heart(g5, g4, P5))
        self.scatter_plots(g4_d, name='unpool')
        g3_d = copy.deepcopy(self.declus_heart(g4, g3, P4))
        self.scatter_plots(g3_d, name='unpool')
        g2_d = copy.deepcopy(self.declus_heart(g3, g2, P3))
        self.scatter_plots(g2_d, name='unpool')
        g1_d = copy.deepcopy(self.declus_heart(g2, g1, P2))
        self.scatter_plots(g1_d, name='unpool')
        g_d = copy.deepcopy(self.declus_heart(g1, g, P1))
        self.scatter_plots(g_d, name='unpool')

        self.save_graph(g, g1, g2, g3, g4, g5, g6, P1, P2, P3, P4, P5, P6)
