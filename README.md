# PIMS-Net: Sea Ice Concentration Estimation via Physical Information-Guided Multi-Source Data Fusion and Spatial Continuity Preservation

This repository provides the PyTorch implementation of **PIMS-Net**, a deep learning framework for high-resolution Arctic sea ice concentration (SIC) estimation from multi-source remote sensing and meteorological observations.

PIMS-Net is designed to retrieve pixel-wise SIC by jointly using:

- **Passive microwave brightness temperatures** from AMSR2;
- **Active microwave SAR backscatter** from Sentinel-1;
- **Meteorological reanalysis variables** from ERA5.

The model follows a physically decoupled multi-branch design. Brightness temperature, SAR, and meteorological variables are encoded by separate ResNet-18 branches, fused through a residual multi-source correction module, and further enhanced by a bidirectional ConvLSTM-based spatial continuity modeling module before SIC reconstruction.

This code accompanies the manuscript:

> **Sea Ice Concentration Estimation via Physical Information-Guided Multi-Source Data Fusion and Spatial Continuity Preservation**  
> Xinyi Liu, Wanxia Deng, Lei Liu, Jinfeng Ding, Xiao Cheng

---

## Overview

Sea ice concentration is a key geophysical variable for Arctic climate monitoring, numerical ocean-sea ice forecasting, and polar navigation. Conventional SIC retrieval algorithms based on passive microwave observations provide wide spatial coverage but are limited by coarse spatial resolution and atmospheric or surface-related uncertainties. SAR imagery provides high-resolution scattering information, while ERA5 variables describe the thermodynamic and atmospheric background. PIMS-Net integrates these complementary sources for high-resolution SIC retrieval.

The main components of PIMS-Net are:

1. **Physically decoupled three-branch encoder**  
   Separate ResNet-18 encoders extract source-specific features from passive microwave brightness temperatures, SAR backscatter, and meteorological variables.

2. **Physical-driven residual correction module (PRCM)**  
   Multi-scale SAR and meteorological features are used to correct the passive microwave feature representation in latent space, preserving the role of radiometric SIC information while introducing scattering and environmental constraints.

3. **Spatial continuity modeling module (SCMM)**  
   Deep fused features are divided into spatial patch sequences and processed using bidirectional ConvLSTM scanning in horizontal and vertical directions. This module enhances spatial continuity and improves the representation of ice edges and transition regions.

4. **Decoder for SIC reconstruction**  
   A U-Net-like decoder progressively upsamples the enhanced latent features and uses skip connections from fused encoder features to generate a pixel-wise SIC map.

---

## Repository Structure

```text
PIMS-Net/
├── main.py                         # Main training entry point
├── config/
│   └── configs.py                  # Variable names, SIC lookup table, and dataset constants
├── model/
│   └── PIMS_Net.py                 # PIMS-Net architecture
├── utils/
│   ├── data_loader.py              # Dataset loader for multi-source NetCDF scenes
│   ├── train_model_regression.py   # Training and validation loop for SIC regression
│   └── functions.py                # Metric utilities, e.g., r2_metric
├── dataset/
│   └── txt/
│       ├── train.txt               # Paths to training scenes
│       ├── val.txt                 # Paths to validation scenes
│       └── test.txt                # Paths to test scenes
└── README.md
```

Please make sure the actual directory names match the imports used in `main.py`:

```python
from utils.data_loader import MyDataset, get_variable_options
from config.configs import CHARTS, SIC_LOOKUP, SCENE_VARIABLES
from utils.train_model_regression import train_model
from model.PIMS_Net import PIMS_Net
```

---

## Requirements

The implementation was developed with PyTorch. A typical environment can be prepared with:

```bash
conda create -n pimsnet python=3.9 -y
conda activate pimsnet
pip install torch torchvision torchaudio
pip install numpy xarray netCDF4 scikit-learn
```

Additional packages may be needed depending on the local NetCDF backend and evaluation scripts.

Core dependencies:

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- xarray
- netCDF4 or h5netcdf

---

## Dataset Preparation

The code is written for the ready-to-train version of the AI4Arctic Sea Ice Dataset. Each scene is expected to be stored as a NetCDF file readable by `xarray.open_dataset`.

The input variables are defined in `config/configs.py`. The default input contains:

### Sentinel-1 SAR and auxiliary spatial variables

```text
nersc_sar_primary
nersc_sar_secondary
```

### AMSR2 brightness temperature channels

```text
btemp_6_9h,  btemp_6_9v
btemp_7_3h,  btemp_7_3v
btemp_10_7h, btemp_10_7v
btemp_18_7h, btemp_18_7v
btemp_23_8h, btemp_23_8v
btemp_36_5h, btemp_36_5v
btemp_89_0h, btemp_89_0v
```

### ERA5 meteorological variables

```text
u10m_rotated
v10m_rotated
t2m
skt
tcwv
tclw
```

### Target

```text
SIC
```

The dataset split is controlled by three text files:

```text
dataset/txt/train.txt
dataset/txt/val.txt
dataset/txt/test.txt
```

Each file should contain one NetCDF scene path per line, for example:

```text
/path/to/scene_001.nc
/path/to/scene_002.nc
/path/to/scene_003.nc
```

In the manuscript experiments, scenes from 2018-2019 were used for training, scenes from 2020 for validation, and scenes from 2021 for testing. Each large scene was cropped into 512 x 512 patches, and the variables were normalized to the range [-1, 1] during preprocessing.

---

## Input Channel Convention

The data loader first reads the SIC chart and input variables from each NetCDF scene, then returns:

```python
data, label, data_name, masks
```

where `data` is the multi-channel input tensor and `label['SIC']` is the SIC reference map.

In the default training script, the input tensor is split as:

```python
bt  = b_x[:, 4:18, :, :]   # 14 AMSR2 brightness temperature channels
sar = b_x[:, :2, :, :]     # 2 Sentinel-1 SAR channels
met = b_x[:, 18:, :, :]    # 6 ERA5 meteorological channels
```

Therefore, the default model is initialized as:

```python
PIMS_Net(tb_ch=14, sar_ch=2, met_ch=6, out_channels=1)
```

The `sar_incidenceangle` and `distance_map` channels are loaded by the current configuration but are not used in the default model input split. Users who want to include these auxiliary variables should modify the channel slicing and the corresponding model input dimensions.

---

## Training

After preparing the dataset text files, train PIMS-Net with:

```bash
python main.py
```

The default training configuration in `main.py` is:

```python
train_options = {
    'lr': 1e-4,
    'epochs': 50,
    'batch_size': 16,
    'patch_size': 512,
    'train_variables': SCENE_VARIABLES,
    'charts': CHARTS,
    'n_classes': {'SIC': SIC_LOOKUP['n_classes']},
    'train_fill_value': 0,
    'class_fill_values': {'SIC': SIC_LOOKUP['mask']},
}
```

The optimizer and scheduler are:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2
)
```

During training, invalid SIC pixels with fill value `255` are masked out. SIC labels are divided by 10 and mapped to the range [0, 1], matching the sigmoid output of the network.

The trained model weights are saved by default to:

```text
model/PIMS_Net/diff_patch_fusion.pth
```

Make sure this directory exists before training:

```bash
mkdir -p model/PIMS_Net
```

---

## Model Architecture

The PIMS-Net architecture is implemented in `model/PIMS_Net.py`.

### Multi-branch encoder

PIMS-Net uses three independent ResNet-18 encoders:

- TB branch for AMSR2 brightness temperatures;
- SAR branch for Sentinel-1 SAR backscatter;
- MET branch for ERA5 meteorological variables.

The first convolution layer of each ResNet-18 encoder is replaced to match the corresponding input channel number.

### Multi-source residual fusion

At each encoder level, features from the three branches are fused by `ConcatSkipFusion`. This module concatenates TB, SAR, and MET features, processes them through convolutional layers, and adds a residual connection from the TB branch.

### Spatial continuity modeling

The deepest fused feature map is divided into a 4 x 4 patch grid. The patches are arranged into two snake-like spatial sequences:

- horizontal row-wise sequence;
- vertical column-wise sequence.

Each sequence is processed by a two-layer bidirectional ConvLSTM. The horizontal and vertical outputs are reassembled, fused, and passed to the decoder.

### Decoder

The decoder uses transposed convolution and skip connections from the fused encoder features. The final layer upsamples the feature map to the original 512 x 512 resolution and applies a sigmoid activation to generate SIC values in [0, 1].

---

## Inference Example

A minimal inference example is shown below:

```python
import torch
from model.PIMS_Net import PIMS_Net

model = PIMS_Net(tb_ch=14, sar_ch=2, met_ch=6, out_channels=1, pretrained=False)
checkpoint = torch.load('model/PIMS_Net/diff_patch_fusion.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Example tensors with 512 x 512 spatial size
tb = torch.randn(1, 14, 512, 512)
sar = torch.randn(1, 2, 512, 512)
met = torch.randn(1, 6, 512, 512)

with torch.no_grad():
    sic = model(tb, sar, met)

print(sic.shape)  # [1, 1, 512, 512]
```

If the checkpoint was trained using `torch.nn.DataParallel`, the keys may contain the prefix `module.`. In that case, remove the prefix before loading or wrap the model with `torch.nn.DataParallel` before loading the checkpoint.

---

## Evaluation Metrics

The manuscript evaluates SIC retrieval from four complementary aspects:

- **MSE**: pixel-wise regression error;
- **R2**: global goodness of fit;
- **SSIM**: spatial structural similarity;
- **Ice-edge MSE**: retrieval error in critical ice-edge regions.

In the reported experiments, PIMS-Net achieved the following test-set performance:

| Model | R2 | MSE | SSIM | Ice-edge MSE |
|---|---:|---:|---:|---:|
| PIMS-Net | 0.935 | 0.0138 | 0.7956 | 0.0874 |
| U-Net | 0.910 | 0.0192 | 0.7196 | 0.1147 |
| DeepLabv3+ | 0.910 | 0.0193 | 0.6976 | 0.1045 |
| Transformer | 0.881 | 0.0254 | 0.7860 | 0.1010 |

---

## Notes

1. The current repository provides the core model, data loader, and training loop. Dataset files and pretrained checkpoints are not included by default.
2. The file `utils/functions.py` should provide the `r2_metric` function used by `train_model_regression.py`.
3. The current code assumes 512 x 512 input patches. If another patch size is used, please check the patch-grid logic in `PIMS_Net.py`.
4. The default task is SIC regression. Although SIC lookup tables contain class information, the provided training loop optimizes MSE loss for continuous SIC estimation.

---

## Citation

If you use this code in your research, please cite the associated manuscript:

```bibtex
@article{liu2026pimsnet,
  title   = {Sea Ice Concentration Estimation via Physical Information-Guided Multi-Source Data Fusion and Spatial Continuity Preservation},
  author  = {Liu, Xinyi and Deng, Wanxia and Liu, Lei and Ding, Jinfeng and Cheng, Xiao},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  year    = {2026},
  note    = {Under review}
}
```

Please update the BibTeX entry after publication.

---

## Contact

For questions about the code or manuscript, please contact:

- Xinyi Liu: liuxinyi20@nudt.edu.cn
- Wanxia Deng: dengwanxia14@nudt.edu.cn
