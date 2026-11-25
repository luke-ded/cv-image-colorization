#!/usr/bin/env python3
###############################################################################
#
# Luke Dederich
# Final Project
# CAP 5415 Computer Vision
# Dr. Rawat
#
###############################################################################
#
# Filename: ConvNet.py
#
# Description: class definition of a U-net-style encoder-decoder structure
# for use as a model for image colorization

# Skip connections via U-Net (Ronneberger et al., 2015), batch normalization 
# from Ioffe & Szegedy (2015).
#
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """U-Net-style encoder-decoder for image colorization.

    Extends torch.nn.Module.

    Args:
        num_bins (int): Number of ab color bins (Q) to predict for each pixel.
    """
    
    def __init__(self, num_bins):
        super(ConvNet, self).__init__()
        
        # Encoding layers
        # Double the channels when halving size
        self.enc_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc_b1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.enc_b2 = nn.BatchNorm2d(32)
        
        self.enc_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc_b3 = nn.BatchNorm2d(64)
        self.enc_conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.enc_b4 = nn.BatchNorm2d(64)
        
        self.enc_conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_b5 = nn.BatchNorm2d(128)
        self.enc_conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_b6 = nn.BatchNorm2d(128)
        
        # Bottleneck
        self.enc_conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_b7 = nn.BatchNorm2d(128)
        self.enc_conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_b8 = nn.BatchNorm2d(128)
        
        # Decoding layers
        self.dec_conv1 = nn.Conv2d(128 + 128, 128, 3, padding=1)
        self.dec_b1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec_b2 = nn.BatchNorm2d(128)
        
        self.dec_conv3 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.dec_b3 = nn.BatchNorm2d(64)
        self.dec_conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_b4 = nn.BatchNorm2d(64)
        
        self.dec_conv5 = nn.Conv2d(64 + 32, 32, 3, padding=1)
        self.dec_b5 = nn.BatchNorm2d(32)
        self.dec_conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.dec_b6 = nn.BatchNorm2d(32)
        
        # Output layer to convert into bin score
        self.out = nn.Conv2d(32, num_bins, 1)


    def forward(self, x):
        """Predicts a color bin distribution for every pixel.

        Args:
            x (torch.Tensor): Normalized L channel, shape (B, 1, H, W). H and W must be divisible by 8.

        Returns:
            torch.Tensor: Log-probabilities over the color bins, shape (B, Q, H, W).
        """
        
        # Encoder layers
        x = F.relu(self.enc_b1(self.enc_conv1(x)))
        x = F.relu(self.enc_b2(self.enc_conv2(x)))
        skip1 = x
        x = F.max_pool2d(x, 2)
        x = F.relu(self.enc_b3(self.enc_conv3(x)))
        x = F.relu(self.enc_b4(self.enc_conv4(x)))
        skip2 = x
        x = F.max_pool2d(x, 2)
        x = F.relu(self.enc_b5(self.enc_conv5(x)))
        x = F.relu(self.enc_b6(self.enc_conv6(x)))
        skip3 = x
        x = F.max_pool2d(x, 2)
        
        # Bottleneck
        x = F.relu(self.enc_b7(self.enc_conv7(x)))
        x = F.relu(self.enc_b8(self.enc_conv8(x)))
        
        # Decoder layers
        
        x = F.interpolate(x, scale_factor=2) # 2x upscale
        x = torch.cat([x, skip3], dim = 1) # Concatenate encoder features (skip connection)
        x = F.relu(self.dec_b1(self.dec_conv1(x)))
        x = F.relu(self.dec_b2(self.dec_conv2(x)))
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, skip2], dim = 1)
        x = F.relu(self.dec_b3(self.dec_conv3(x)))
        x = F.relu(self.dec_b4(self.dec_conv4(x)))
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, skip1], dim = 1)
        x = F.relu(self.dec_b5(self.dec_conv5(x)))
        x = F.relu(self.dec_b6(self.dec_conv6(x)))
        
        
        log_probs = F.log_softmax(self.out(x), dim=1)
        
        return log_probs