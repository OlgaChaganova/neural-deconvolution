from __future__ import print_function, absolute_import
import os
from operator import itemgetter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
from sklearn.feature_extraction import image
from skimage.color import rgb2gray
from torch.utils.data import Dataset

from src.data.make_kernel import kernel_sim_spline


class ToTensor(object):
    def __call__(self, sample):
        y, k, kt = sample['y'], sample['k'], sample['kt']
        img_ch_num = len(y.shape)
        if img_ch_num == 2:
            y = y
        elif img_ch_num == 3:
            y = y.transpose((2, 0, 1))
        return torch.from_numpy(y).float(), \
            torch.from_numpy(k.reshape(1, k.shape[0], k.shape[1])).float(), \
            torch.from_numpy(kt.reshape(1, k.shape[0], k.shape[1])).float()


class BlurryImageDataset(Dataset):
    """Blur image dataset"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_name_list = [
            name for name in os.listdir(self.root_dir)
            if os.path.isfile(os.path.join(self.root_dir, name)) and name.endswith('.mat')
        ]
        self.file_name_list.sort()
        self.TensorConverter = ToTensor()

    def __len__(self):
        return len([name for name in os.listdir(self.root_dir) \
                    if os.path.isfile(os.path.join(self.root_dir, name)) and name.endswith('.mat') ])

    def __getitem__(self, idx):
        """get .mat file"""
        mat_name = self.file_name_list[idx]
        sample = sio.loadmat(os.path.join(self.root_dir, mat_name))
        if self.transform:
            sample = self.transform(sample)

        return self.TensorConverter(sample), mat_name


class BlurryImageDatasetOnTheFly(Dataset):
    def __init__(self,
                 file_name_list: list,
                 k_size=41,
                 sp_size=[11, 16, 21, 26, 31],
                 num_spl_ctrl=[3, 4, 5, 6],
                 patch_size=256,
                 is_rgb=True,
                 max_num_images=None):
        self.k_size = k_size
        self.sp_size = sp_size
        self.num_spl_ctrl = num_spl_ctrl
        self.patch_size = 256
        self.is_rgb = is_rgb
        self.rksize = [
            11,
        ]

        self.file_name_list = sorted(file_name_list)

        if max_num_images is not None and max_num_images < len(
                self.file_name_list):
            self.file_name_list = self.file_name_list[:max_num_images]

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        sp_size = int(np.random.choice(self.sp_size))
        num_spl_ctrl = int(np.random.choice(self.num_spl_ctrl))
        # print(sp_size, num_spl_ctrl)
        k = kernel_sim_spline(sp_size, self.k_size, num_spl_ctrl, 1)
        k = np.reshape(k, [1, 1, self.k_size, self.k_size])

        img_name = self.file_name_list[idx]
        sample = plt.imread(img_name)

        if sample.shape[0] < self.patch_size or sample.shape[
                1] < self.patch_size:
            return self.__getitem__((idx - 1) % (self.__len__()))
        patches = image.extract_patches_2d(sample,
                                           [self.patch_size, self.patch_size],
                                           max_patches=1)
        sample = patches[0, ...]
        sample = sample.astype(np.float32) / 255.0
        if not self.is_rgb:
            sample = rgb2gray(sample)
            sample = np.expand_dims(sample, 2)
        sample = np.expand_dims(np.transpose(sample, [2, 0, 1]), 1)
        sample = torch.from_numpy(sample.astype(np.float32))  # n x c x w x h
        hks = (self.k_size) // 2

        with torch.no_grad():
            k = torch.from_numpy(k)
            y = torch.nn.functional.conv2d(sample, k)
            nl = np.random.uniform(0.003, 0.015)
            y = y + nl * torch.randn_like(y)
            y = torch.clamp(y * 255.0, 0, 255)
            y = y.type(torch.ByteTensor)
            y = y.float() / 255.0
            # y = torch.nn.functional.pad(y, (hks, hks, hks, hks),
            #                             mode='replicate')
            y = y.squeeze(1)
            x_gt = sample.squeeze(1)[:, hks:(-hks), hks:(-hks)]
            k = k.squeeze(0)
            kt = torch.flip(k, [1, 2])

        return y, x_gt, k, kt

########## ADDED ###############    
def get_data(data_dir, is_silent=False):
    train_dir = Path(os.path.join(data_dir, 'train'))
    test_dir = Path(os.path.join(data_dir, 'test'))

    train_files = list(train_dir.rglob('*.jpg'))
    test_files = list(test_dir.rglob('*.jpg'))
        
    ids = np.random.permutation(len(test_files))
    N = len(test_files) // 2

    valid_files = list(itemgetter(*ids[:N])(test_files))
    test_files = list(itemgetter(*ids[N:])(test_files))
    
    if not is_silent:
        print('Files are loaded.')
        print('Train size: ', len(train_files), '\t', 'Valid size: ', len(valid_files), '\t', 'Test size: ', len(test_files))
    
    return train_files, valid_files, test_files 
###################################