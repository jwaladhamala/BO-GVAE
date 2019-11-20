import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.read import read_ply
import numpy as np
import scipy.io
from torch_geometric.data import Data


class HeartNetSubset(InMemoryDataset):
    url = 'NO URL'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 mfree=None):
        super(HeartNetSubset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        self.mfree = mfree

    @property
    def raw_file_names(self):
        return 'testing_ir_ic_1230.mat'

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download testing_ir_ic_???? from {} and '
            'move it to {}'.format(self.url, self.raw_dir))

    def process(self):
        # extract data from the matlab files
        #path = osp.join(self.raw_dir, 'MPI-FAUST', 'training', 'registrations')
        filename = 'testing_ir_ic_' + str(self.mfree) + '.mat'
        path = osp.join(self.raw_dir, filename)

        matFiles = scipy.io.loadmat(path,squeeze_me=True,struct_as_record=False)
        corMfree = matFiles['corMfree']
        dataset = matFiles['param_list_ir_e']
        label_aha = matFiles['label_aha_ir_e']
        label_loc = matFiles['label_loc_ir_e']
        label_size = matFiles['label_size_ir_e']
        label_lrv = matFiles['label_lrv_ir_e']

        num_nodes, tot_data  = dataset.shape
        tot_data = 1000
        data_list = []
        for i in range(tot_data):
            pos = torch.from_numpy(corMfree).float() #torch.tensor(corMfree,dtype=torch.float)
            x = torch.from_numpy(dataset[:,[i]]).float()#torch.tensor(dataset[:,[i]],dtype=torch.float)
            y = torch.from_numpy(label_aha[[i]]).float()#torch.tensor(label_aha[[i]],dtype=torch.float)
            data = Data(pos = pos, x = x, y = y) # geometry, features, and label
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)            
            data_list.append(data)
        
        np.random.seed(45)
        rnd_idx  = np.random.permutation(tot_data)

        train_split, test_split = rnd_idx[:int(0.8 *tot_data)], rnd_idx[int(0.8 *tot_data):]
        data_list_train = [data_list[i] for i in train_split]
        data_list_test = [data_list[i] for i in test_split]
        

        torch.save(self.collate(data_list_train), self.processed_paths[0])
        torch.save(self.collate(data_list_test), self.processed_paths[1])
        
        
