import os.path as osp

import scipy.io
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data


class HeartGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. 
    """

    def __init__(self,
                 root,
                 num_meshfree=None,
                 mesh_graph=None,
                 train=True,
                 label_type='size'):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'raw')
        filename = '_ir_ic_' + str(num_meshfree) + '.mat'
        # print(label_type)
        if train:
            filename = 'training' + filename
            self.data_path = osp.join(self.raw_dir, filename)
            matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
            corMfree = matFiles['corMfree']
            tissue_params = matFiles['param_list_ir_t']
            label_aha = matFiles['label_aha_ir_t']
            label_loc = matFiles['label_loc_ir_t']
            label_size = matFiles['label_size_ir_t']
            label_lrv = matFiles['label_lrv_ir_t']
            N = tissue_params.shape[1]
            # print('filename {}'.format(filename))
            # print('total data size: {}'.format(N))
            # N = 2560#int(N/5)
        else:
            filename = 'testing' + filename
            self.data_path = osp.join(self.raw_dir, filename)
            matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
            corMfree = matFiles['corMfree']
            tissue_params = matFiles['param_list_ir_e']
            label_aha = matFiles['label_aha_ir_e']
            label_loc = matFiles['label_loc_ir_e']
            label_size = matFiles['label_size_ir_e']
            label_lrv = matFiles['label_lrv_ir_e']
            N = tissue_params.shape[1]

        #N = int(N / 10)

        if label_type == 'size':
            self.label = torch.from_numpy(label_size[0:N]).float()
        else:
            self.label = torch.from_numpy(label_aha[0:N]).float()
        self.graph = mesh_graph
        self.datax = torch.from_numpy(tissue_params[:, 0:N]).float()
        self.corMfree = corMfree
        print('final data size: {}'.format(self.datax.shape[1]))

    def getCorMfree(self):
        return self.corMfree

    def __len__(self):
        return (self.datax.shape[1])

    def __getitem__(self, idx):

        x = self.datax[:, [idx]]  # torch.tensor(dataset[:,[i]],dtype=torch.float)
        y = self.label[[idx]]  # torch.tensor(label_aha[[i]],dtype=torch.float)

        sample = Data(x=x,
                      y=y,
                      edge_index=self.graph.edge_index,
                      edge_attr=self.graph.edge_attr,
                      pos=self.graph.pos)
        return sample
