#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main training script for PIMS-Net."""
__authors__ = 'Xinyi Liu, Wanxia Deng, Lei Liu, Jinfeng Ding, Xiao Cheng'
__copyright__ = ['National University of Defense Technology School of Meteorology and Oceanology']
__contact__ = ['liuxinyi20@nudt.edu.cn', 'dengwanxia14@nudt.edu.cn']
__version__ = '1.0.0'
__date__ = '2026-04-30'

# -- Third-party modules -- #
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# -- Proprietary modules -- #
from utils.data_loader import MyDataset,get_variable_options
from config.configs import CHARTS, SIC_LOOKUP, SCENE_VARIABLES
from utils.train_model_regression import train_model
# from utils.train_model_diff_task import train_model
# from utils.imbalance import train_model
# from model.unet import UNet
from model.PIMS_Net import PIMS_Net
# from model.PIMS_Net_diff_patch import PIMS_Net
# from model.PIMS_Net_diff_task import PIMS_Net
# from model.PIMS_Net_diff_patch_fusion import PIMS_Net
# from model.deeplabv3plus import DeepLabV3Plus
# from model.transformer import Transformer
# from model.PIMS_Net_wo_PRCM import PIMS_Net_wo_PRCM
# from model.PIMS_Net_wo_SCMM import PIMS_Net_wo_SCMM

# -- Device configuration -- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Training configuration -- #
train_options = {
    'lr': 1e-4,                      # Learning rate
    'epochs': 50,                    # Number of training epochs
    'batch_size': 16,                # Batch size
    'patch_size': 512,               # Input patch size (height/width)
    'train_variables': SCENE_VARIABLES,   # Names of all input variables
    'charts': CHARTS,                # Target variable names (e.g., SIC)
    'n_classes': {'SIC': SIC_LOOKUP['n_classes']},   # Number of classes for segmentation
    'train_fill_value': 0,           # Fill value for missing training data
    'class_fill_values': {'SIC': SIC_LOOKUP['mask']},  # Fill value for target masks

    # -- U-Net Options (if used) -- #
    'unet_conv_filters': [16, 32, 64, 64],   # Number of filters in U-Net encoder
    'conv_kernel_size': (3, 3),              # Convolution kernel size
    'conv_stride_rate': (1, 1),              # Convolution stride
    'conv_dilation_rate': (1, 1),            # Dilation rate
    'conv_padding': (1, 1),                  # Padding size
    'conv_padding_style': 'zeros',           # Padding style
}

get_variable_options = get_variable_options(train_options)

# -- DataLoaders -- #
# Training dataset
train_dataset = MyDataset(r"dataset/txt/train.txt", train_options['charts'], train_options)
# Validation dataset
val_dataset = MyDataset(r"dataset/txt/val.txt", train_options['charts'], train_options)
# Test dataset
test_dataset = MyDataset(r"dataset/txt/test.txt", train_options['charts'], train_options)

# Data loaders with batching and parallel loading
train_loader = Data.DataLoader(train_dataset, batch_size=train_options['batch_size'],
                               shuffle=True, num_workers=4, pin_memory=True)
val_loader = Data.DataLoader(val_dataset, batch_size=train_options['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)
test_loader = Data.DataLoader(test_dataset, batch_size=train_options['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

# -- Model initialization -- #
sic_seg_model = PIMS_Net(tb_ch=14, sar_ch=2, met_ch=6, out_channels=1).to(device)
# sic_seg_model = PIMS_Net(
#             tb_ch=14,
#             sar_ch=2,
#             met_ch=6,
#             pretrained=False,
#             patch_order_h='diag_main',
#             patch_order_v='diag_anti',
#             random_seed=42
#         ).to(device)
# sic_seg_model = UNet(in_channel=22, pretrained=False).to(device)
# sic_seg_model = DeepLabV3Plus(in_channels=22, pretrained=False).to(device)
# sic_seg_model = Transformer(in_channels=22).to(device)
# sic_seg_model = PIMS_Net_wo_PRCM(in_channels=22, out_channels=1, pretrained=True).to(device)
# sic_seg_model = PIMS_Net_wo_SCMM(tb_ch=14, sar_ch=2, met_ch=6, out_channels=1).to(device)
# sic_seg_model = PIMS_Net(
#             tb_ch=14,
#             sar_ch=2,
#             met_ch=6,
#             out_channels=11,
#             pretrained=False,
#             task_type='classification'
#         ).to(device)

if torch.cuda.device_count() > 1:
    sic_seg_model = nn.DataParallel(sic_seg_model)

# -- Optimizer and scheduler -- #
optimizer = optim.AdamW(sic_seg_model.parameters(), lr=train_options['lr'], weight_decay=1e-4)

# Cosine annealing with warm restarts (restart every 5 epochs, double period after each restart)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# -- Main entry point: train and save model -- #
if __name__ == '__main__':
    # Train the model
    sic_seg_model = train_model(sic_seg_model, optimizer, train_loader, val_loader, train_options, scheduler)

    # Save trained weights
    model_name = 'model/PIMS_Net/diff_patch_fusion' + '.pth'
    torch.save(sic_seg_model.state_dict(), model_name)