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

EXTRA_TRAIN_CLASSES = [
    "/m/mountain", "/m/mountain_snowy", "/c/cliff", "/c/creek", "/r/river",
    "/i/islet", "/s/swamp", "/m/marsh", "/p/pond", "/o/orchard",
    "/v/vineyard", "/c/corn_field", "/f/field/cultivated", "/f/forest_path",
    "/f/forest_road", "/t/tree_farm", "/d/desert/vegetation", "/r/rock_arch",
    "/l/lawn", "/f/field_road",
]

ALL_CLASSES = CLASSES + EXTRA_TRAIN_CLASSES

def select_indices(places, test_prop=0.2, seed=1, chroma=None, min_chroma=2, classes=CLASSES):
    class_ids = {places.class_to_idx[name]: [] for name in classes}
    
    for index, target in enumerate(places.targets):
        if target in class_ids:
            # Filter out near grayscale images
            if chroma is not None and chroma[index] < min_chroma:
                continue
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

def to_rgb(L, ab):
    lab = torch.cat([(L + 1) * 50, ab], dim=1)
    lab = lab.permute(0,2 , 3, 1).cpu().numpy()
    return color.lab2rgb(lab)

class PlacesDataset(torch.utils.data.Dataset):
    def __init__(self, places, indices, image_size, train):
        self.places = places
        self.indices = indices
        
        if train:
            # Transform with flips to help avoid overfitting
            self.transform=transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
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
        