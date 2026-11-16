#!/usr/bin/env python3

# Luke Dederich
# Final Project
# CAP 5415 Computer Vision
# Dr. Rawat

# Add skip connections?
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
        Defines a Autoencoder Structure
        Extends torch.nn.Module class
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
        self.dec_conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec_b1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec_b2 = nn.BatchNorm2d(128)
        
        self.dec_conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_b3 = nn.BatchNorm2d(64)
        self.dec_conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_b4 = nn.BatchNorm2d(64)
        
        self.dec_conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec_b5 = nn.BatchNorm2d(32)
        self.dec_conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.dec_b6 = nn.BatchNorm2d(32)
        
        # Output layer to convert into bin number
        self.out = nn.Conv2d(32, num_bins, 1)

    def forward(self, x):
        # Encoder layers
        x = F.relu(self.enc_b1(self.enc_conv1(x)))
        x = F.relu(self.enc_b2(self.enc_conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.enc_b3(self.enc_conv3(x)))
        x = F.relu(self.enc_b4(self.enc_conv4(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.enc_b5(self.enc_conv5(x)))
        x = F.relu(self.enc_b6(self.enc_conv6(x)))
        x = F.max_pool2d(x, 2)
        
        # Bottleneck
        x = F.relu(self.enc_b7(self.enc_conv7(x)))
        x = F.relu(self.enc_b8(self.enc_conv8(x)))
        
        # Decoder layers
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.dec_b1(self.dec_conv1(x)))
        x = F.relu(self.dec_b2(self.dec_conv2(x)))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.dec_b3(self.dec_conv3(x)))
        x = F.relu(self.dec_b4(self.dec_conv4(x)))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.dec_b5(self.dec_conv5(x)))
        x = F.relu(self.dec_b6(self.dec_conv6(x)))
        
        
        probs = F.log_softmax(self.out(x), dim=1)
        
        return probs