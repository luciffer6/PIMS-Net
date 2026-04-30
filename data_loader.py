#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data loader module for PIMS-Net: reads multi-modal data (TB, SAR, MET) and labels."""
__authors__ = 'Xinyi Liu, Wanxia Deng, Lei Liu, Jinfeng Ding, Xiao Cheng'
__copyright__ = ['National University of Defense Technology School of Meteorology and Oceanology']
__contact__ = ['liuxinyi20@nudt.edu.cn', 'dengwanxia14@nudt.edu.cn']
__version__ = '1.0.0'
__date__ = '2026-04-30'


# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


def read_path(root):
    """
        Read file paths from a text file.

        Parameters
        ----------
        root : str
            Path to a text file containing one file path per line.

        Returns
        -------
        list of str
            List of file paths.
        """
    data = np.loadtxt(root, dtype=str)
    n = len(data)
    datas = [None] * n
    for i, fname in enumerate(data):
        datas[i] = fname
    return datas


class MyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading multi-modal remote sensing data (TB, SAR, MET)
    and corresponding sea ice concentration labels.
    """

    def __init__(self, data_root,charts,options):
        self.data_root = data_root
        data_list = read_path(root=data_root)
        self.data_list = data_list
        self.charts = charts
        self.options= options

    def data_prepare(self, scene):
        """
        Perform random cropping in scene.

        Parameters
        ----------
        scene :
            Xarray dataset; a scene from ASID3 ready-to-train challenge dataset.

        Returns
        -------
        patch :
            Numpy array with shape (len(train_variables), patch_height, patch_width). None if empty patch.
        """
        patch = np.zeros((len(self.options['full_variables']) + len(self.options['amsrenv_variables']),
                          self.options['patch_size'], self.options['patch_size']))

        patch[0:len(self.options['full_variables']), :, :] = scene[self.options['full_variables']].to_array()

        patch[len(self.options['full_variables']):, :, :] = scene[self.options['amsrenv_variables']].to_array()

        return patch


    def prep_dataset(self, patches):
        """
        Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W].

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        # Convert training data to tensor.
        x = torch.from_numpy(patches[len(self.charts):]).type(torch.float)

        # Store charts in y dictionary.
        y = {}
        for idx, chart in enumerate(self.charts):
            y[chart] = torch.from_numpy(patches[idx]).type(torch.long)

        return x, y


    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        data_name = os.path.basename(self.data_list[idx])  # 获取文件名部分

        scene = xr.open_dataset(data_path)
        scene_patch = self.data_prepare(scene)
        data, label = self.prep_dataset(patches=scene_patch)
        masks = {}
        for chart in self.options['charts']:
            masks[chart] = (label[chart] == self.options['class_fill_values'][chart]).squeeze()

        return data, label, data_name, masks

    def __len__(self):
        return len(self.data_list)


def get_variable_options(train_options: dict):
    """
    Get amsr and env grid options, crop shape and upsampling shape.

    Parameters
    ----------
    train_options: dict
        Dictionary with training options.

    Returns
    -------
    train_options: dict
        Updated with amsrenv options.
    """
    train_options['sar_variables'] = [variable for variable in train_options['train_variables'] \
                                      if 'sar' in variable or 'map' in variable]
    train_options['full_variables'] = np.hstack((train_options['charts'], train_options['sar_variables']))
    train_options['amsrenv_variables'] = [variable for variable in train_options['train_variables'] \
                                          if 'sar' not in variable and 'map' not in variable]

    return train_options