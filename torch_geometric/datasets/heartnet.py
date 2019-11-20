import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.read import read_ply
import numpy as np
import scipy.io
from torch_geometric.data import Data


class HeartNet(InMemoryDataset):
    """
    This dataset is similar to the datasets defined by pytorch geometric. It can be used to 
    read raw heart data and make graphs out of them. It will also save pickle files for training
    and testing. It will make a graph for each data point so it is computationally expensive.
    """
    url = 'NO URL'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(HeartNet, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        if root[-5,:] == 'case2':
            self.pdim = 1373
        elif self.root[-5,:] == 'case1':
            self.pdim = 1239
        else:
            self.pdim = 1230
        self.fname = 'training_ir_ic_' + str(self.pdim) +'.mat'

    @property
    def raw_file_names(self):
        return self.fname

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download training_ir_ic_1230 from {} and '
            'move it to {}'.format(self.url, self.raw_dir))

    def process(self):
        # extract data from the matlab files
        #path = osp.join(self.raw_dir, 'MPI-FAUST', 'training', 'registrations')
        path = osp.join(self.raw_dir,self.fname)
        
        


        matFiles = scipy.io.loadmat(path,squeeze_me=True,struct_as_record=False)
        corMfree = matFiles['corMfree']
        dataset = matFiles['param_list_t']
        label_aha = matFiles['label_aha_t']
        label_loc = matFiles['label_loc_t']
        label_size = matFiles['label_size_t']
        label_lrv = matFiles['label_lrv_t']

        num_nodes, tot_data  = dataset.shape

        data_list = []
        tot_data = 1000
        for i in range(tot_data):
            pos = torch.tensor(corMfree)
            x = torch.tensor(dataset[:,[i]])
            y = torch.tensor(label_aha[[i]])
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
        
        
