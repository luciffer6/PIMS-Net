# PIMS-Net: Multi-modal Fusion Network for Sea Ice Concentration Prediction

This repository contains the official implementation of **PIMS-Net**, a multi-modal deep learning model for sea ice concentration (SIC) prediction. The model fuses thermal (TB), SAR, and meteorological (MET) data using a ResNet encoder, ConcatSkipFusion, ConvBiLSTM, and a decoder.

## Key Features

- Multi-modal fusion of TB, SAR, and MET inputs
- ResNet18-based encoders with skip connections
- ConcatSkipFusion modules for feature integration
- Bidirectional ConvLSTM for spatial scanning
- Regression output for SIC (0–100%)
- Supports training, validation, and testing on the ASID dataset

## Dependencies

The code requires Python 3.9+ and the following packages (tested versions shown):
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
xarray>=2022.10.0
h5netcdf
scikit-learn>=1.2.0
Pillow>=9.0.0
tqdm>=4.64.0


Install all dependencies using:

```bash
pip install -r requirements.txt

## Data Preparation
