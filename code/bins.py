#!/usr/bin/env python3
###############################################################################

# Luke Dederich
# Final Project
# CAP 5415 Computer Vision
# Dr. Rawat

###############################################################################
#
# Filename: bins.py
#
# Description: run individually to set up bins by finding which bins to use, 
# the center of each bin, and which colors are common/rare. Also contains bin 
# related helper functions.
#
# Method via Zhang, Isola & Efros, "Colorful Image Colorization" (ECCV 2016)
#
###############################################################################

import torch

from torch.utils.data import DataLoader
from torchvision import datasets

# Custom file
import data

# Quantize ab space into a 22x22 grid of 10x10 cells covering [-110, 110]
AB_MIN = -110
GRID = 10
N_CELLS = 22
N_CELLS_SQ = N_CELLS * N_CELLS


def ab_to_cell(ab):
    """Maps ab values to their grid cell index.

    Args:
        ab (torch.Tensor): Color channels, shape (B, 2, H, W).

    Returns:
        torch.Tensor: Grid cell index per pixel, shape (B, H, W), dtype long, 
            values in [0, N_CELLS_SQ - 1].
    """
    
    index = torch.floor((ab - AB_MIN) / GRID).long().clamp(0, N_CELLS - 1)
    a_index = index[:, 0]
    b_index = index[:, 1]
    cell = a_index * N_CELLS + b_index
    
    return cell


def encode(ab, cell_to_bin):
    """Returns ground truth bin number for pixel

    Args:
        ab (torch.Tensor): Color channels, shape (B, 2, H, W).
        cell_to_bin (torch.Tensor): Lookup table from grid cell to bin index,
            shape (N_CELLS_SQ,), dtype long, -1 for unused cells.

    Returns:
        torch.Tensor: Bin index per pixel, shape (B, H, W), values in [0, Q - 1], 
            or -1 if the cell is unused.
    """
    
    cell = ab_to_cell(ab)
    
    return cell_to_bin[cell]


def decode(log_probs, centers):
    """Converts model output to ab colors by taking each pixel's most likely bin.

    Args:
        log_probs (torch.Tensor): Log-probabilities from the model, shape (B, Q, H, W).
        centers (torch.Tensor): ab value at the center of each bin, shape (Q, 2).

    Returns:
        torch.Tensor: Predicted ab channels, shape (B, 2, H, W).
    """
    
    index = log_probs.argmax(dim=1)
    
    ab = centers[index]
    ab = ab.permute(0, 3, 1 ,2) # Re order channels
    
    return ab


def decode_annealed(log_probs, centers, T=0.7):
    """Converts model output to ab colors using the annealed mean.

    Sharpens each pixel's bin distribution with temperature T, then takes the
    probability-weighted average of the bin centers.

    Args:
        log_probs (torch.Tensor): Log-probabilities from the model, shape (B, Q, H, W).
        centers (torch.Tensor): ab value at the center of each bin, shape (Q, 2).
        T (float, optional): Softmax temperature. Lower values are more vivid,
            higher are more dull. Defaults to 0.7.

    Returns:
        torch.Tensor: Predicted ab channels, shape (B, 2, H, W).
    """
    
    probs = torch.nn.functional.softmax(log_probs / T, dim=1)
    
    return torch.einsum("bqhw,qc->bchw", probs, centers)
    
        
if __name__ == "__main__":        
    # Load datasets for training and testing
    places = datasets.Places365("./data", split="train-standard", small=True, download=True)
    chroma = torch.load("./data/chroma.pt")["train-standard"]
    train_indices, test_indices = data.select_indices(places, chroma=chroma, classes=data.ALL_CLASSES)
        
    train_set = data.PlacesDataset(places, train_indices, 64, train=False)
        
    train_loader = DataLoader(train_set, batch_size = 100, 
                                shuffle=True, num_workers=4, persistent_workers=True)
    
    # Count the number of training pixels in each grid cell
    counts = torch.zeros(N_CELLS_SQ, dtype=torch.long)
    
    for L, ab in train_loader:
        cell = ab_to_cell(ab)
        counts += torch.bincount(cell.flatten(), minlength=N_CELLS_SQ)
    
    # Keep only cells that occur in training data 
    kept = counts > 0
    Q = int(kept.sum())
    
    # Number the used cells
    cell_to_bin = torch.full((N_CELLS_SQ,), -1, dtype=torch.long)
    cell_to_bin[kept] = torch.arange(Q)
        
        
    # Calculate bin centers (what each bin's color is)
    all_cells = torch.arange(N_CELLS_SQ)
        
    a_index = all_cells // N_CELLS
    b_index = all_cells % N_CELLS
        
    center_a = AB_MIN + (GRID * a_index) + (GRID / 2)
    center_b = AB_MIN + (GRID * b_index) + (GRID / 2)
        
    all_centers = torch.stack([center_a, center_b], dim=1) # (N_CELLS_SQ, 2)
    centers = all_centers[kept]
    
    # Rebalance weights. Rare, vivid colors get higher weights to prevent dullness.
    lambda_value = 0.5
    p = counts[kept] / counts.sum()
    w = 1 / ((1 - lambda_value) * p + (lambda_value / Q))
    w = w / (p * w).sum()
    
    torch.save({
        "cell_to_bin": cell_to_bin,          # (484,) long
        "centers":     centers.float(),      # (Q, 2)
        "weights":     w.float(),            # (Q,)
        "counts":      counts,               # (484,)
    }, "./data/bins.pt")
    