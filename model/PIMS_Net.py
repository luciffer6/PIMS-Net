#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PIMS-Net model."""
__authors__ = 'Xinyi Liu, Wanxia Deng, Lei Liu, Jinfeng Ding, Xiao Cheng'
__copyright__ = ['National University of Defense Technology School of Meteorology and Oceanology']
__contact__ = ['liuxinyi20@nudt.edu.cn', 'dengwanxia14@nudt.edu.cn']
__version__ = '1.0.0'
__date__ = '2026-04-30'

# -- File info -- #
# This file implements the PIMS-Net model which fuses thermal, SAR and meteorological data
# using a ResNet encoder, ConcatSkipFusion, ConvBiLSTM, and a decoder for sea ice concentration.

# -- Third-party modules -- #
import torch
import torch.nn as nn
from torchvision import models


class ConcatSkipFusion(nn.Module):
    """
    Fusion module that concatenates three feature maps (TB, SAR, MET) and applies two conv layers.
    A residual connection from the TB branch is added after projection if dimensions differ.
    """
    def __init__(self, dim_tb, dim_sar, dim_met, out_dim=None):
        """
        Args:
            dim_tb (int): Channels of TB feature map.
            dim_sar (int): Channels of SAR feature map.
            dim_met (int): Channels of MET feature map.
            out_dim (int, optional): Output channels. Defaults to dim_tb.
        """
        super().__init__()
        if out_dim is None:
            out_dim = dim_tb

        total_channels = dim_tb + dim_sar + dim_met

        # Two convolutional layers with batch norm and ReLU
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # Projection layer for residual connection if channel mismatch
        if out_dim != dim_tb:
            self.residual_proj = nn.Conv2d(dim_tb, out_dim, kernel_size=1)
        else:
            self.residual_proj = None

    def forward(self, tb_feat, sar_feat, met_feat):
        """
        Forward pass.
        Args:
            tb_feat (torch.Tensor): TB features [B, C_tb, H, W]
            sar_feat (torch.Tensor): SAR features [B, C_sar, H, W]
            met_feat (torch.Tensor): MET features [B, C_met, H, W]
        Returns:
            torch.Tensor: Fused features [B, out_dim, H, W]
        """
        fused = torch.cat([tb_feat, sar_feat, met_feat], dim=1)
        fused_out = self.fusion_conv(fused)
        if self.residual_proj is not None:
            tb_feat = self.residual_proj(tb_feat)
        return fused_out + tb_feat


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatiotemporal processing."""
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        """
        Args:
            input_dim (int): Number of input channels.
            hidden_dim (int): Number of hidden state channels.
            kernel_size (int): Size of the convolutional kernel.
            bias (bool): Whether to use bias in conv.
        """
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        # Convolution that produces 4 * hidden_dim (for i, f, o, g gates)
        self.input_conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                                    kernel_size, padding=padding, bias=bias)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_cur, c_cur):
        """
        Forward pass for one time step.
        Args:
            x (torch.Tensor): Input at current time [B, input_dim, H, W]
            h_cur (torch.Tensor): Current hidden state [B, hidden_dim, H, W]
            c_cur (torch.Tensor): Current cell state [B, hidden_dim, H, W]
        Returns:
            h_next, c_next: Updated hidden and cell states.
        """
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.input_conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvBiLSTM(nn.Module):
    """Bidirectional convolutional LSTM with multiple layers."""
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1):
        """
        Args:
            input_dim (int): Input feature channels.
            hidden_dim (int): LSTM hidden channels.
            kernel_size (int): Kernel size for ConvLSTM.
            num_layers (int): Number of stacked BiLSTM layers.
        """
        super(ConvBiLSTM, self).__init__()
        self.num_layers = num_layers
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()

        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim * 2
            self.forward_cells.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
            self.backward_cells.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))

    def forward(self, input_seq):
        """
        Process input sequence in both directions.
        Args:
            input_seq (torch.Tensor): Sequence of patches [B, T, C, H, W]
        Returns:
            torch.Tensor: Concatenated forward+backward outputs [B, T, 2*hidden_dim, H, W]
        """
        B, T, C, H, W = input_seq.size()
        device = input_seq.device

        for i in range(self.num_layers):
            # Initialize forward and backward states
            h_f, c_f = (torch.zeros(B, self.forward_cells[i].hidden_dim, H, W, device=device),
                        torch.zeros(B, self.forward_cells[i].hidden_dim, H, W, device=device))
            h_b, c_b = (torch.zeros(B, self.backward_cells[i].hidden_dim, H, W, device=device),
                        torch.zeros(B, self.backward_cells[i].hidden_dim, H, W, device=device))

            output_f, output_b = [], []
            # Forward pass
            for t in range(T):
                h_f, c_f = self.forward_cells[i](input_seq[:, t], h_f, c_f)
                output_f.append(h_f)
            # Backward pass
            for t in reversed(range(T)):
                h_b, c_b = self.backward_cells[i](input_seq[:, t], h_b, c_b)
                output_b.append(h_b)
            output_b.reverse()  # restore temporal order

            # Concatenate forward and backward outputs along channel dimension
            input_seq = torch.cat([torch.stack(output_f, dim=1),
                                   torch.stack(output_b, dim=1)], dim=2)

        return input_seq


class DecoderBlock(nn.Module):
    """Upsampling block with skip connection."""
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Input channels from lower level.
            out_channels (int): Output channels after upconv and convs.
        """
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x, skip):
        """
        Args:
            x (torch.Tensor): Feature from lower level [B, in_channels, H, W]
            skip (torch.Tensor): Skip connection from encoder [B, out_channels, 2H, 2W]
        Returns:
            torch.Tensor: Upsampled and refined features.
        """
        x = self.upconv(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class PIMS_Net(nn.Module):
    """
    Main PIMS-Net model: Multi-modal fusion with ResNet encoders, ConcatSkipFusion,
    ConvBiLSTM for spatial scanning, and a decoder.
    """
    def __init__(self, tb_ch, sar_ch, met_ch, out_channels=1, pretrained=True,
                 en_patch_grid=4, lstm_hidden_dim=256):
        """
        Args:
            tb_ch (int): Number of TB input channels.
            sar_ch (int): Number of SAR input channels.
            met_ch (int): Number of MET input channels.
            out_channels (int): Output channels (1 for binary SIC).
            pretrained (bool): Use pretrained ResNet18 weights.
            en_patch_grid (int): Grid size for patch extraction (e.g., 4 -> 4x4 patches).
            lstm_hidden_dim (int): Hidden dimension for ConvBiLSTM.
        """
        super(PIMS_Net, self).__init__()
        self.en_patch_grid = en_patch_grid
        self.en_patch_size = 512 // en_patch_grid   # assuming input size 512

        # ---------- ResNet Encoders for each modality ----------
        # TB encoder
        resnet18 = models.resnet18(pretrained=pretrained)
        resnet18.conv1 = nn.Conv2d(tb_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc1 = nn.Sequential(resnet18.conv1, resnet18.bn1, resnet18.relu)
        self.enc2 = nn.Sequential(resnet18.maxpool, resnet18.layer1)
        self.enc3 = resnet18.layer2
        self.enc4 = resnet18.layer3
        self.enc5 = resnet18.layer4

        # SAR encoder
        resnet18_sar = models.resnet18(pretrained=pretrained)
        resnet18_sar.conv1 = nn.Conv2d(sar_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc1_sar = nn.Sequential(resnet18_sar.conv1, resnet18_sar.bn1, resnet18_sar.relu)
        self.enc2_sar = nn.Sequential(resnet18_sar.maxpool, resnet18_sar.layer1)
        self.enc3_sar = resnet18_sar.layer2
        self.enc4_sar = resnet18_sar.layer3
        self.enc5_sar = resnet18_sar.layer4

        # MET encoder
        resnet18_met = models.resnet18(pretrained=pretrained)
        resnet18_met.conv1 = nn.Conv2d(met_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc1_met = nn.Sequential(resnet18_met.conv1, resnet18_met.bn1, resnet18_met.relu)
        self.enc2_met = nn.Sequential(resnet18_met.maxpool, resnet18_met.layer1)
        self.enc3_met = resnet18_met.layer2
        self.enc4_met = resnet18_met.layer3
        self.enc5_met = resnet18_met.layer4

        # ---------- Multi-modal Fusion modules ----------
        self.fusion_e1 = ConcatSkipFusion(dim_tb=64, dim_sar=64, dim_met=64, out_dim=64)
        self.fusion_e2 = ConcatSkipFusion(dim_tb=64, dim_sar=64, dim_met=64, out_dim=64)
        self.fusion_e3 = ConcatSkipFusion(dim_tb=128, dim_sar=128, dim_met=128, out_dim=128)
        self.fusion_e4 = ConcatSkipFusion(dim_tb=256, dim_sar=256, dim_met=256, out_dim=256)
        self.fusion_e5 = ConcatSkipFusion(dim_tb=512, dim_sar=512, dim_met=512, out_dim=512)

        # ---------- Bidirectional ConvLSTM for spatial scanning ----------
        self.bilstm = ConvBiLSTM(512, lstm_hidden_dim, num_layers=2)

        # Boundary fusion after reassembling patches
        self.boundary_fusion = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv to reduce dual-scan concatenation back to 512 channels
        self.conv1x1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True)

        # ---------- Decoder with skip connections ----------
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 64)

        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, tb, sar, met):
        """
        Forward pass.
        Args:
            tb (torch.Tensor): TB input [B, tb_ch, H, W], H=W=512 recommended.
            sar (torch.Tensor): SAR input [B, sar_ch, H, W]
            met (torch.Tensor): MET input [B, met_ch, H, W]
        Returns:
            torch.Tensor: Sea ice concentration map [B, out_channels, H, W] after sigmoid.
        """
        B, C, H, W = tb.shape

        # ---------- Encoder forward for each modality ----------
        # TB branch
        f1 = self.enc1(tb)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)

        # SAR branch
        f1_sar = self.enc1_sar(sar)
        f2_sar = self.enc2_sar(f1_sar)
        f3_sar = self.enc3_sar(f2_sar)
        f4_sar = self.enc4_sar(f3_sar)
        f5_sar = self.enc5_sar(f4_sar)

        # MET branch
        f1_met = self.enc1_met(met)
        f2_met = self.enc2_met(f1_met)
        f3_met = self.enc3_met(f2_met)
        f4_met = self.enc4_met(f3_met)
        f5_met = self.enc5_met(f4_met)

        # ---------- Fusion at each encoder level ----------
        f_e1 = self.fusion_e1(f1, f1_sar, f1_met)   # 64 channels
        f_e2 = self.fusion_e2(f2, f2_sar, f2_met)   # 64 channels
        f_e3 = self.fusion_e3(f3, f3_sar, f3_met)   # 128 channels
        f_e4 = self.fusion_e4(f4, f4_sar, f4_met)   # 256 channels
        f_e5 = self.fusion_e5(f5, f5_sar, f5_met)   # 512 channels

        # ---------- Patchify and BiLSTM scanning (horizontal and vertical) ----------
        patch_h, patch_w = f_e5.shape[2] // self.en_patch_grid, f_e5.shape[3] // self.en_patch_grid

        # Horizontal snake scanning (row-major, alternating direction)
        patches_h = []
        for i in range(self.en_patch_grid):
            cols = range(self.en_patch_grid) if i % 2 == 0 else range(self.en_patch_grid - 1, -1, -1)
            for j in cols:
                patch = f_e5[:, :, i * patch_h: (i + 1) * patch_h, j * patch_w: (j + 1) * patch_w]
                patches_h.append(patch)
        patch_seq_h = torch.stack(patches_h, dim=1)   # [B, N_patches, 512, patch_h, patch_w]

        # Vertical snake scanning (column-major, alternating direction)
        patches_v = []
        for j in range(self.en_patch_grid):
            rows = range(self.en_patch_grid) if j % 2 == 0 else range(self.en_patch_grid - 1, -1, -1)
            for i in rows:
                patch = f_e5[:, :, i * patch_h: (i + 1) * patch_h, j * patch_w: (j + 1) * patch_w]
                patches_v.append(patch)
        patch_seq_v = torch.stack(patches_v, dim=1)

        # Process each sequence through ConvBiLSTM
        lstm_out_h = self.bilstm(patch_seq_h)   # [B, N_patches, 2*hidden, patch_h, patch_w]
        lstm_out_v = self.bilstm(patch_seq_v)

        # ---------- Reassemble patches back to full feature map ----------
        recon_feat_h = torch.zeros(B, lstm_out_h.size(2), f_e5.shape[2], f_e5.shape[3], device=tb.device)
        idx = 0
        for i in range(self.en_patch_grid):
            cols = range(self.en_patch_grid) if i % 2 == 0 else range(self.en_patch_grid - 1, -1, -1)
            for j in cols:
                recon_feat_h[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w] = lstm_out_h[:, idx]
                idx += 1
        recon_feat_h = self.boundary_fusion(recon_feat_h)

        recon_feat_v = torch.zeros(B, lstm_out_v.size(2), f_e5.shape[2], f_e5.shape[3], device=tb.device)
        idx = 0
        for j in range(self.en_patch_grid):
            rows = range(self.en_patch_grid) if j % 2 == 0 else range(self.en_patch_grid - 1, -1, -1)
            for i in rows:
                recon_feat_v[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w] = lstm_out_v[:, idx]
                idx += 1
        recon_feat_v = self.boundary_fusion(recon_feat_v)

        # Concatenate horizontal and vertical scanned features and reduce channels
        recon_feat = torch.cat((recon_feat_h, recon_feat_v), dim=1)
        recon_feat = self.conv1x1(recon_feat)

        # ---------- Decoder with skip connections from fused encoder features ----------
        x = self.dec4(recon_feat, f_e4)
        x = self.dec3(x, f_e3)
        x = self.dec2(x, f_e2)
        x = self.dec1(x, f_e1)   # note: skip from first fusion level
        x = self.final_up(x)

        # Apply sigmoid for binary probability output
        output = torch.sigmoid(x)
        return output


if __name__ == "__main__":
    # Simple test to verify model forward pass
    B, H, W = 1, 512, 512
    tb_ch = 14
    sar_ch = 2
    met_ch = 6

    model = PIMS_Net(tb_ch=tb_ch, sar_ch=sar_ch, met_ch=met_ch, pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f} M")

    tb = torch.randn(B, tb_ch, H, W)
    sar = torch.randn(B, sar_ch, H, W)
    met = torch.randn(B, met_ch, H, W)

    out = model(tb, sar, met)
    print("SIC shape:", out.shape)