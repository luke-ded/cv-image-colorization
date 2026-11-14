import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ConvNet import ConvNet
import argparse


def main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() # Add MPS for macos gpu

    # Set proper device based on cuda/mps availability 
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    print("Torch device selected: ", device)
        
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)

    # Define loss function. NLLLoss, combined with the log-softmax layer applied 
    # in the CNN, creates CrossEntropyLoss. Torch's built in CrossEntropyLoss
    # automatically applies softmax, which would result in two applications 
    criterion = nn.NLLLoss()
        
    # Define optimizer function.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate, momentum=0.9, weight_decay=5e-4)
        
    # Define scheduler function
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLAGS.num_epochs)
        
    # Create transformations to apply to each data sample 
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        
    # Transform with random crops and flips to help avoid overfitting
    train_transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 2),
        transforms.ToTensor(),
        normalize
    ])
        
    # Unmodified transform for test
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
        
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.Places365('./data/', train=True, download=True,
                    transform=train_transform)
    dataset2 = datasets.Places365('./data/', train=False,
                    transform=test_transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4, persistent_workers=True)



if __name__ == "__main__":
    # Set parameters for CNN Exercise
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-2.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=50,
                        help='Batch size. Must divide evenly into the dataset sizes.')
        
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
        
    main(FLAGS)