import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import data

if __name__ == "__main__":
    chroma = {}
    
    for split in ["train-standard", "val"]:
        places = datasets.Places365("./data", split=split, small=True)
        indices, _ = data.select_indices(places, test_prop=0)
        dataset = data.PlacesDataset(places, indices, 64, train=False)
        loader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=4)
        
        values = []
        for L, ab in loader:
            values.append(ab.pow(2).sum(dim=1).sqrt().mean(dim=(1, 2)))
        values = torch.cat(values)
        
        chroma[split] = dict(zip(indices, values.tolist()))
        
    torch.save(chroma, "data/chroma.pt")