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

text

Install all dependencies using:

```bash
pip install -r requirements.txt
Data Preparation
Organize your data as NetCDF files with the following variables (see config/configs.py for exact names):

SAR: nersc_sar_primary, nersc_sar_secondary, sar_incidenceangle, distance_map

TB: btemp_6_9h, btemp_6_9v, ..., btemp_89_0v

MET: u10m_rotated, v10m_rotated, t2m, skt, tcwv, tclw

Prepare text files listing the paths to training, validation, and test .nc files, e.g.:

text
dataset/txt/train.txt
dataset/txt/val.txt
dataset/txt/test.txt
Each line should contain the full path to a NetCDF file.

Training
Modify main.py to set your data paths and hyperparameters (batch size, learning rate, epochs, etc.). Then run:

bash
python main.py
Training logs will be printed to the console. The best model (lowest validation MSE) will be saved as model/PIMS_Net/diff_patch_fusion.pth.

Testing / Inference
Load the trained model and run on test data (implementation can be added in main.py or a separate script). Example:

python
from model.PIMS_Net_diff_patch_fusion import PIMS_Net

model = PIMS_Net(tb_ch=14, sar_ch=2, met_ch=6, pretrained=False)
model.load_state_dict(torch.load('model/PIMS_Net/diff_patch_fusion.pth'))
model.eval()

# Prepare inputs (bt, sar, met) as torch tensors of shape (B, C, H, W)
with torch.no_grad():
    output = model(bt, sar, met)  # output shape: (B, 1, H, W), values in [0,1]
File Structure
text
.
├── config/configs.py                # Dataset constants and lookup tables
├── dataset/txt/                     # Text files with data paths
├── model/
│   └── PIMS_Net_diff_patch_fusion.py   # Model definition
├── utils/
│   ├── data_loader.py               # MyDataset and variable options
│   ├── functions.py                 # Metrics (e.g., r2_metric)
│   └── train_model_regression.py    # Training loop
├── main.py                          # Main training script
└── requirements.txt                 # Dependencies
Citation
If you use this code in your research, please cite the original paper:

bibtex
@article{liu2026pimsnet,
  title={PIMS-Net: Multi-modal Fusion Network for Sea Ice Concentration Prediction},
  author={Liu, Xinyi and Deng, Wanxia and Liu, Lei and Ding, Jinfeng and Cheng, Xiao},
  year={2026}
}
License
This project is licensed under the MIT License – see the LICENSE file for details.
