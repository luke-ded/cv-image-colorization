#!/usr/bin/env python3
###############################################################################

# Luke Dederich
# Final Project
# CAP 5415 Computer Vision
# Dr. Rawat

###############################################################################
## The goal: set up bins by finding which bins to use, the center of each bin, 
## and which colors are common/rare
###############################################################################

import data
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

AB_MIN = -110
GRID = 10
N_CELLS = 22
N_CELLS_SQ = N_CELLS * N_CELLS

def ab_to_cell(ab):
    index = torch.floor((ab - AB_MIN) / GRID).long().clamp(0, N_CELLS - 1)
    a_index = index[:, 0]
    b_index = index[:, 1]
    cell = a_index * N_CELLS + b_index
    
    return cell

# Returns ground truth bin number for pixel
def encode(ab, cell_to_bin):
    cell = ab_to_cell(ab)  # (B, H, W), values 0–483
    return cell_to_bin[cell] # (B, H, W), values 0–205 or -1

# Returns model prediction from bin to color
def decode(log_probs, centers):
    index = log_probs.argmax(dim=1) # (B, H, W), values 0–205
    
    ab = centers[index]
    ab = ab.permute(0, 3, 1 ,2) # Re order channels
    
    return ab

# Returns model prediction from bin to color
def decode_annealed(log_probs, centers, T=0.7):
    probs = torch.nn.functional.softmax(log_probs / T, dim=1)
    return torch.einsum("bqhw,qc->bchw", probs, centers)
        
if __name__ == "__main__":        
    # Load datasets for training and testing
    places = datasets.Places365("./data", split="train-standard", small=True, download=True)
    chroma = torch.load("./data/chroma.pt")["train-standard"]
    train_indices, test_indices = data.select_indices(places, chroma=chroma)
        
    train_set = data.PlacesDataset(places, train_indices, 64, train=False)
        
    train_loader = DataLoader(train_set, batch_size = 100, 
                                shuffle=True, num_workers=4, persistent_workers=True)
    
    # Count the number of training pixels in each grid cell
    counts = torch.zeros(N_CELLS_SQ, dtype=torch.long)
    
    for L, ab in train_loader:
        cell = ab_to_cell(ab)
        counts += torch.bincount(cell.flatten(), minlength=N_CELLS_SQ)
        
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
    
    # Rebalance weights
    p = counts[kept] / counts.sum()
    w = 1 / ((1 - 0.5) * p + (0.5 / Q))
    w = w / (p * w).sum()
    
    torch.save({
        "cell_to_bin": cell_to_bin,          # (484,) long
        "centers":     centers.float(),      # (Q, 2)
        "weights":     w.float(),            # (Q,)
        "counts":      counts,               # (484,)
    }, "./data/bins.pt")
    