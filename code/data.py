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
# Filename: data.py
#
# Description: contains the class definition wrapper for the torch Places365
# dataset with transforms and Lab handling. Also contains data helper functions
# and lists of hand-selected outdoor classes.
#
###############################################################################

import torch
import random

from torchvision import transforms
from skimage import color

# Selected classes of outdoor scenes with varying colors
CLASSES = [
    # Blue: water / coast
    "/b/beach",
    "/c/coast",
    "/o/ocean",
    "/l/lagoon",
    "/l/lake/natural",
    "/w/waterfall",
    # Green: vegetation
    "/f/forest/broadleaf",
    "/r/rainforest",
    "/b/bamboo_forest",
    "/f/field/wild",
    "/p/pasture",
    "/r/rice_paddy",
    "/b/botanical_garden",
    # Yellow / gold: fields, sand
    "/w/wheat_field",
    "/h/hayfield",
    "/d/desert/sand",
    # Red / brown / orange: rock, terrain
    "/c/canyon",
    "/b/badlands",
    "/b/butte",
    "/v/valley",
]

# Extra classes for extended training
EXTRA_TRAIN_CLASSES = [
    "/m/mountain",
    "/m/mountain_snowy",
    "/c/cliff",
    "/c/creek",
    "/r/river",
    "/i/islet",
    "/s/swamp",
    "/m/marsh",
    "/p/pond",
    "/o/orchard",
    "/v/vineyard",
    "/c/corn_field",
    "/f/field/cultivated",
    "/f/forest_path",
    "/f/forest_road",
    "/t/tree_farm",
    "/d/desert/vegetation",
    "/r/rock_arch",
    "/l/lawn",
    "/f/field_road",
]

# All selected classes
ALL_CLASSES = CLASSES + EXTRA_TRAIN_CLASSES


def select_indices(
    places, test_prop=0.2, seed=1, chroma=None, min_chroma=2.0, classes=CLASSES
):
    """Splits the images of the given classes into train and test indices.

    Args:
        places (torchvision.datasets.Places365): Dataset to select from.
        test_prop (float, optional): Proportion of data to use for testing. Defaults to 0.2.
        seed (int, optional): Seed for random number generator. Defaults to 1.
        chroma (dict[int, float], optional): Chroma values of places images. If None, no filtering.
            Defaults to None.
        min_chroma (float, optional): Low chroma image removal threshold. Defaults to 2.0.
        classes (list[str], optional): Classes to select indices from. Defaults to CLASSES.

    Returns:
        tuple[list[int], list[int]]: (train_indices, test_indices) into places.
    """

    class_ids = {places.class_to_idx[name]: [] for name in classes}

    for index, target in enumerate(places.targets):
        if target in class_ids:
            # Filter out near grayscale images (value 2 chosen via visual inspection)
            if chroma is not None and chroma[index] < min_chroma:
                continue
            class_ids[target].append(index)

    # Seeded split per class, so every script (chroma, bins, train) gets the same split
    rng = random.Random(seed)
    train_idx = []
    test_idx = []

    for image_class in class_ids.keys():

        rng.shuffle(class_ids[image_class])
        n_test = int(len(class_ids[image_class]) * test_prop)

        train_idx.extend(class_ids[image_class][n_test:])
        test_idx.extend(class_ids[image_class][:n_test])

    return train_idx, test_idx


def to_lab(rgb):
    """Converts an RGB image to CIELab.

    Args:
        rgb (torch.Tensor): rgb image. Shape (3, H, W), values in [0, 1].

    Returns:
        torch.Tensor: CIELab image. Shape (3, H, W), L in [0, 100], a and b in ~[-110, 110].
    """

    return torch.from_numpy(color.rgb2lab(rgb.numpy(), channel_axis=0))


def to_rgb(L, ab):
    """Converts a batch of normalized L and ab channels to RGB images.

    Args:
        L (torch.Tensor): Normalized lightness. Shape (B, 1, H, W), values in [-1, 1].
        ab (torch.Tensor): Color channels. Shape (B, 2, H, W).

    Returns:
        np.ndarray: RGB images, shape (B, H, W, 3), values in [0, 1].
    """

    lab = torch.cat([(L + 1) * 50, ab], dim=1)
    lab = lab.permute(0, 2, 3, 1).cpu().numpy()

    return color.lab2rgb(lab)


class PlacesDataset(torch.utils.data.Dataset):
    """Places365 wrapper that returns normalized L and ab channels.

    Extends torch.utils.data.Dataset.

    Args:
        places (torchvision.datasets.Places365): Underlying dataset.
        indices (list[int]): Indices into places to use.
        image_size (int): Height/width images are resized or cropped to.
        train (bool): If True, applies random crop and flip augmentation.

    Each item is a tuple (L, ab): L has shape (1, H, W) with values in [-1, 1],
    and ab has shape (2, H, W).
    """

    def __init__(self, places, indices, image_size, train):
        self.places = places
        self.indices = indices

        if train:
            # Transform with flips and crops to help avoid overfitting
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            # Unmodified transform for test
            self.transform = transforms.Compose(
                [transforms.Resize(image_size), transforms.ToTensor()]
            )

    def __len__(self):
        """Returns the number of images.

        Returns:
            int: number of images.
        """

        return len(self.indices)

    def __getitem__(self, index):
        """Returns (L, ab) for the image at index.

        Args:
            index (int): Position in this dataset's indices list.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (L, ab).
        """

        image = self.places[self.indices[index]][0]
        normalized_rgb = self.transform(image)
        lab = to_lab(normalized_rgb)

        L = lab[0:1]
        ab = lab[1:3]

        # Normalize L to [-1, 1]
        L = (L / 50) - 1

        return L, ab
