#!/usr/bin/env python3
###############################################################################

# Luke Dederich
# Final Project
# CAP 5415 Computer Vision
# Dr. Rawat

###############################################################################
#
# Filename: visualize.py
#
# Description: helper functions for data visualization
#
###############################################################################

import torch
import matplotlib
import matplotlib.pyplot as plt

# Custom files
import bins
import data

matplotlib.use("Agg")


# Show matrix image
def show_plot(matrix, axis, title=""):
    """Shows matrix image using matplotlib.

    Args:
        matrix (np.ndarray): Image, shape (H, W, 3) or (H, W).
        axis (matplotlib.axes.Axes): Axis to draw the image on.
        title (str, optional): Plot title. Defaults to "".
    """

    axis.imshow(matrix, cmap="gray")
    if title:
        axis.set_title(title)
    axis.axis("off")


def save_samples(model, device, test_set, centers, path, n=6):
    """Saves a grid of test samples with input, model output, and ground truth.

    Args:
        model (ConvNet): The model to sample from. Should already be in correct device.
        device (torch.device): Device the model is on, values 'cuda' or 'mps' or 'cpu'.
        test_set (PlacesDataset): Test dataset.
        centers (torch.Tensor): ab value at the center of each bin, shape (Q, 2).
        path (str): Path to save figure.
        n (int, optional): Number of samples to show. Defaults to 6.
    """

    model.eval()

    # Evenly spaced indices so samples span classes
    picks = [i * len(test_set) // n for i in range(n)]

    L = torch.stack([test_set[i][0] for i in picks]).to(device)
    ab = torch.stack([test_set[i][1] for i in picks]).to(device)

    with torch.no_grad():
        predicted_ab = bins.decode_annealed(model(L), centers)

    grayscale = data.to_rgb(L, torch.zeros_like(ab))
    predicted = data.to_rgb(L, predicted_ab)
    truth = data.to_rgb(L, ab)

    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    for row in range(n):
        show_plot(grayscale[row], axes[row, 0], "Input" if row == 0 else "")
        show_plot(predicted[row], axes[row, 1], "Predicted" if row == 0 else "")
        show_plot(truth[row], axes[row, 2], "Truth" if row == 0 else "")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
