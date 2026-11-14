#!/usr/bin/env python3

# Luke Dederich
# Programming Assignment 3
# CAP 5415 Computer Vision
# Dr. Rawat

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
        Defines a Autoencoder Structure
        Extends torch.nn.Module class
    """    
    def __init__(self, mode):
        """
            Initializes the Autoencoder structure based on mode
            mode (int): 1-2, determines step to run
        """
        super(ConvNet, self).__init__()
        
        # Define layers based on mode/step and select the forward pass function based on mode for the ConvNet.
        if mode == 1:
            # 28 * 28 = 784
            self.enc_fc1 = nn.Linear(784, 256)
            self.enc_fc2 = nn.Linear(256, 128)
            self.dec_fc1 = nn.Linear(128, 256)
            self.dec_fc2 = nn.Linear(256, 784)

            self.forward = self.model_1
        elif mode == 2:
            self.enc_conv1 = nn.Conv2d(1, 32, 3, 1, 1)
            self.enc_conv2 = nn.Conv2d(32, 16, 3, 1, 1)
            self.dec_conv1 = nn.Conv2d(16, 16, 3, 1, 1)
            self.dec_conv2 = nn.Conv2d(16, 32, 3, 1, 1)
            self.dec_conv3 = nn.Conv2d(32, 1, 3, 1, 1)
            
            self.forward = self.model_2
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-2")
            exit(0)
        
        
    def model_1(self, X):
        # ======================================================================
        # Fully conencted neural network
        
        # Flatten for FC layers
        X = X.view(-1, self.num_flat_features(X))
        
        # Encoder layers
        X = F.relu(self.enc_fc1(X))
        X = F.relu(self.enc_fc2(X))
        
        # Decoder layers
        X = F.relu(self.dec_fc1(X))
        X = torch.sigmoid(self.dec_fc2(X))
        
        # Transform output into 28 pixel images for MNIST
        fcl = X.view(-1, 1, 28, 28)
        
        return fcl

    def model_2(self, X):
        # ======================================================================
        # Convolutional neural network

        # Encoder layers
        X = F.max_pool2d(F.relu(self.enc_conv1(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.enc_conv2(X)), (2, 2))
        
        # Decoder layers
        X = F.interpolate(F.relu(self.dec_conv1(X)), scale_factor=2)
        X = F.interpolate(F.relu(self.dec_conv2(X)), scale_factor=2)
        
        
        fcl = self.dec_conv3(X)
        return fcl

    
    # Computes number of features flattened; from class notes
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features