import os.path as osp

import scipy.io
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data


class MySimpleHeartDataset(Dataset):
    """
    A dataset of Data objects with only features x and labels y.
    To be used to train a VAE with fully connected layers.
    """

    def __init__(self, root, num_meshfree, train=True, label_type='size'):
        # super(MySimpleHeartDataset, self).__init__()
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'raw')
        filename = '_ir_ic_' + str(num_meshfree) + '.mat'

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
        #  labels can be either size of the scar or location of the scar
        if label_type == 'size':
            self.label = torch.from_numpy(label_size[0:N]).float()
        else:
            self.label = torch.from_numpy(label_aha[0:N]).float()

        self.datax = torch.from_numpy(tissue_params[:, 0:N]).float()
        self.corMfree = corMfree

        print('filename: {}, Number of data points used: {}'.format(filename, self.datax.shape[1]))

    def getCorMfree(self):
        """Get the number of meshfree node in the heart
        """
        return self.corMfree

    def __len__(self):
        return (self.datax.shape[1])

    def __getitem__(self, idx):
        x = self.datax[:, [idx]]  # torch.tensor(dataset[:,[i]],dtype=torch.float)
        y = self.label[[idx]]  # torch.tensor(label_aha[[i]],dtype=torch.float)

        sample = Data(x=x, y=y)
        return sample
