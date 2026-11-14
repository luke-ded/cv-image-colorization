import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ConvNet import ConvNet 
import argparse
import numpy as np 
import csv
import time
from datetime import timedelta
import data
    
def main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() # Add MPS for macos gpu

    # Set proper device based on cuda/mps availability 
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    print("Torch device selected: ", device)
    
    # Load datasets for training and testing
    places = datasets.Places365("./data", split="val", small=True, download=True)
    train_indices, test_indices = data.select_indices(places)
    
    train_set = data.PlacesDataset(places, train_indices, FLAGS.image_size, train=True)
    test_set = data.PlacesDataset(places, test_indices, FLAGS.image_size, train=False)
    
    train_loader = DataLoader(train_set, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4, persistent_workers=True)
     
    L, ab = next(iter(train_loader))
    print(L.shape, ab.shape)   # expect [50, 1, 64, 64] and [50, 2, 64, 64]
    return
   
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)

    # Define loss function
    criterion = nn.NLLLoss()
        
    # Define optimizer function.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate, momentum=0.9, weight_decay=5e-4)
        
    # Define scheduler function
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLAGS.num_epochs)
    
    best_accuracy = 0.0
        
    # Set up data file for report analysis
    data_file = open(f"data/output/data-{FLAGS.mode}-{FLAGS.learning_rate}-{FLAGS.num_epochs}-{FLAGS.batch_size}.csv", mode='w', newline="", encoding="utf-8")
    writer = csv.writer(data_file)
    writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        
    # Record start time
    start_time = time.perf_counter()
        
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print(f"\nEpoch {epoch}")
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
            
        scheduler.step()
            
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            
        # Save data to analysis file
        writer.writerow([epoch, train_loss, train_accuracy, test_loss, test_accuracy])
        
    # Record end and total time
    end_time = time.perf_counter()
    proc_time = end_time - start_time
    td_duration = timedelta(seconds=proc_time)
        
    print("accuracy is {:2.2f}".format(100. * best_accuracy))
        
    print(f"Processing took {td_duration} total, {td_duration / FLAGS.num_epochs} per epoch")
        
    print("Training and evaluation finished")
        
    data_file.close()



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
    parser.add_argument('--image_size', type=int, default=64,
                        help='Height/width images are resized to.')

        
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
        
    main(FLAGS)