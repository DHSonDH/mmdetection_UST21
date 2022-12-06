import os
import glob
import argparse
import numpy as np
import xarray as xr

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

data_path = './DaesanDataset'

class NIA_Dataset(Dataset):
    def __init__(self,
                 args: argparse,
                 base_dir: str = data_path,
                 mode: str = 'train'):

        self.args = args
        self.base_dir = base_dir
        self.mode = mode

        # Transform array to tensor
        if mode != 'test':
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.RandomResizedCrop(args.resize),
                 ])
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor()
                 ])

        # self.value_vars = ['lag00_u','lag00_v','lag00_temp','lag00_sst','lag00_qff',
        #               'lag00_rh','lag00_vis','lag00_ASTD','lag00_Td','lag00_temp-Td',
        #               'lag00_sst-Td','hour','month', 
        #               'h_msharp_sel','h_msharp_sel2',
        #               'h_msharp_sel3','h_msharp_sel22', 'h_msharp_sel33','h_msharp_sel222',
        #               'h_msharp_sel333', 'psnr0', 'psnr1', 'psnr2', 'px_intensity1',
        #               'px_intensity2']  # 25개
        self.value_vars = ['lag00_u','lag00_v','lag00_temp','lag00_sst','lag00_qff',
                      'lag00_rh','lag00_vis','lag00_ASTD','lag00_Td','lag00_temp-Td',
                      'lag00_sst-Td','hour','month', 'h_msharp_sel','h_msharp_sel2',
                      'h_msharp_sel3','h_msharp_sel22', 'h_msharp_sel33','h_msharp_sel222',
                      'h_msharp_sel333', 'psnr1', 'psnr2']  # 22개
        
        data_dir = r'D:\2021\multimodal_NIA\NIA_dataset2'
        data_port_nc = os.path.join(data_dir, f'hjh_{self.mode}.nc')
        self.nc_read = xr.open_dataset(data_port_nc)
        self.site_onehot = np.load(f'{data_dir}/hjh_{self.mode}.npy')
        self.y = self.nc_read.label
        # self.img = self.nc_read.rgb
        self.img = np.swapaxes(self.nc_read.rgb, 1,3)
        
        Table = self.nc_read.drop_vars(['label','rgb'])
        self.table_features = np.zeros((len(self.nc_read.label), len(self.value_vars)))
        for i, var in enumerate(self.value_vars):
            A = Table[var].values
            self.table_features[:,i] = A
        
    def __getitem__(self, idx):
        
        # 라벨 처리
        label = int(self.y[idx])
        target = self.load_target(label)
        
        
        # CCTV 이미지 처리
        img_features = self.img[idx].to_numpy()
        image_x = self.transform(img_features)
        
        # observation 처리
        table_features = self.table_features[idx]
        # table_features = np.stack([self.table.isel(filename = idx)[var].values 
        #                                 for var in self.value_vars])
        
        # one-hot feature 추가 
        table_features_onehot = np.concatenate((table_features,
                                                self.site_onehot[idx]))
        obs_x = self.load_structure(table_features_onehot)
        
        return {'structure_x': obs_x,
                'image_x': image_x,
                'target_y': target
                # 'filename': self.y.filename
                }


    def __len__(self):

        return len(self.nc_read.label)

        
    @staticmethod
    def load_structure(data: dict = None):
        # obs_x = data['structure_x']
        obs_x = data
        obs_x = torch.FloatTensor(obs_x)

        return obs_x


    @staticmethod
    def load_target(data: dict = None):
        
        # print(np.unique(data))
        # target = data['label']
        target = data
        target = np.expand_dims(target, axis=0)
        target = torch.LongTensor(target)

        return target



if __name__=='__main__':
    from types import SimpleNamespace
    
    args  = SimpleNamespace()
    args.resize = 256
    
    dset = NIA_Dataset(args = args,  base_dir = '', mode='test')
    
    print('structure_x shape is: ', dset[0]['structure_x'].shape)
    print('image_x shape is: ', dset[0]['image_x'].shape)
    print('target_y shape is: ', dset[0]['target_y'].shape)
    
    # (1, 33), (3 , 256, 256), [1]
    # 자동으로 축 하나 추가하는데, 33이여야할듯. 
