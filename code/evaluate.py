#!/usr/bin/env python3
###############################################################################

# Luke Dederich
# Final Project
# CAP 5415 Computer Vision
# Dr. Rawat

###############################################################################
#
# Filename: evaluate.py
#
# Description: evaluates a model outputting PSNR, SSIM, and Colorfulness 
# metrics
#
###############################################################################

import argparse
import warnings
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from ConvNet import ConvNet

# Custom files
import data
import bins


def colorfulness(image):
    """Computes the colorfulness metric of Hasler & Süsstrunk (2003).

    Args:
        image (np.ndarray): RGB image, shape (H, W, 3), values in [0, 1].

    Returns:
        float: Colorfulness score (0 for grayscale; higher is more colorful).
    """
    
    image = image * 255 # Un normalize values
    
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    
    std_dev = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    
    return std_dev + (0.3 * mean)
    

def main(FLAGS):
    """Runs evaluation on the model specified by --model_path.

    Args:
        FLAGS (argparse.Namespace): Parsed argparse arguments.
    """
    
    # Ignore lab2rgb warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() # Add MPS for macos gpu

    # Set proper device based on cuda/mps availability 
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    print("Torch device selected: ", device)
    
    # Load bins
    bins_data = torch.load("./data/bins.pt")
    centers = bins_data["centers"].to(device)
    weights = bins_data["weights"].to(device)
    Q = len(weights)

    # Initialize the model and send to device 
    model = ConvNet(Q).to(device)
    
    model.load_state_dict(torch.load(FLAGS.model_path, map_location=device))
    model.eval()
    
    # Val set is used for test    
    places = datasets.Places365("./data", split="val", small=True, download=True)
    chroma = torch.load("./data/chroma.pt")["val"]
    val_indices, _ = data.select_indices(places, test_prop=0, chroma=chroma)
            
    val_set = data.PlacesDataset(places, val_indices, FLAGS.image_size, train=False)
            
    val_loader = DataLoader(val_set, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4, persistent_workers=True)
    
    results = {name: {"psnr": [], "ssim": [], "color": []} for name in ["gray", "predicted", "truth"]}
    
    with torch.no_grad():
        for L, ab in val_loader:
            L, ab = L.to(device), ab.to(device)
            predicted_ab = bins.decode_annealed(model(L), centers)
            
            images = {
                "gray": data.to_rgb(L, torch.zeros_like(ab)), # Grayscale baseline
                "predicted": data.to_rgb(L, predicted_ab),
                "truth": data.to_rgb(L, ab)
            }
            
            for i in range(len(L)):
                truth = images["truth"][i]
                for name, batch in images.items():
                    image = batch[i]
                    
                    if name != "truth":
                        # We cant calculate these metrics for truth
                        results[name]["psnr"].append(peak_signal_noise_ratio(truth, image, data_range=1.0))
                        results[name]["ssim"].append(structural_similarity(truth, image, channel_axis=-1, data_range=1.0)) # SSIM: Wang et al. (2004)
                    results[name]["color"].append(colorfulness(image))

    # Display output table
    print("Output\t\tPSNR\tSSIM\tColorfulness")
    for name, r in results.items():
        psnr = np.mean(r["psnr"]) if r["psnr"] else float("nan")
        ssim = np.mean(r["ssim"]) if r["ssim"] else float("nan")
        print(f"{name:10s}\t{psnr:.2f}\t{ssim:.4f}\t{np.mean(r['color']):.2f}")
    print()
            
       
if __name__ == "__main__":
    # Set parameters for evaluation
    parser = argparse.ArgumentParser('Evaluates Colorization Performance of a Model.')
    parser.add_argument('--batch_size',
                        type=int, default=50,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Height/width images are resized to.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model weights for evaluation.')
      
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)