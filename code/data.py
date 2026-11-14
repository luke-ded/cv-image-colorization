from torchvision import datasets, transforms
import torch
import random

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

def select_indices(places, test_prop=0.2, seed=1):
    class_ids = {places.class_to_idx[name]: [] for name in CLASSES}
    
    for index, target in enumerate(places.targets):
        if target in class_ids.keys():
            class_ids[target].append(index)
            
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
    return torch.from_numpy(color.rgb2lab(rgb.numpy(), channel_axis=0))

def to_rgb(lab):
    return torch.from_numpy(color.lab2rgb(lab.numpy(), channel_axis=0))

class PlacesDataset(torch.utils.data.Dataset):
    def __init__(self, places, indices, image_size, train):
        self.places = places
        self.indices = indices
        
        if train:
            # Transform with flips to help avoid overfitting
            self.transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:      
            # Unmodified transform for test
            self.transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        image = self.places[self.indices[index]][0]
        normalized_rgb = self.transform(image)
        lab = to_lab(normalized_rgb)
        
        L = lab[0:1]
        ab = lab[1:3]
        
        # Center L for better training
        L = (L / 50) - 1
        
        return L, ab
        