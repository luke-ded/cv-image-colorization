#!/usr/bin/env python3

# Luke Dederich
# Final Project
# CAP 5415 Computer Vision
# Dr. Rawat

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
import os
import math
from datetime import timedelta
import data, bins, visualize

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, cell_to_bin):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'mps' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    total = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        L, ab = batch_sample # L is data
        
        # Push data/label to correct device
        L, ab = L.to(device), ab.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(L)
        
        # ======================================================================
        # Compute loss based on criterion
        target = bins.encode(ab, cell_to_bin)
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
            
        # Count correct predictions overall 
        correct += torch.count_nonzero(pred == target.view_as(pred))
        total += (target != -1).sum().item()

    correct = correct.item()    
    train_loss = float(np.mean(losses))
    train_acc = correct / total
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        float(np.mean(losses)), correct, total,
        100. * train_acc))
    return train_loss, train_acc
    

def test(model, device, test_loader, criterion, cell_to_bin):
    '''
    Tests the model.
    model: The model to evaluate. Should already be in correct device.
    device: 'cuda' or 'mps' or 'cpu'.
    test_loader: dataloader for test samples.
    criterion: used to compute loss for prediction and target 
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    total = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            L, ab = sample
            L, ab = L.to(device), ab.to(device)
            
            # Predict for data by doing forward pass
            output = model(L)
            
            # ======================================================================
            # Compute loss based on same criterion as training
            target = bins.encode(ab, cell_to_bin)
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
                
            # Count correct predictions overall 
            correct += torch.count_nonzero(pred == target.view_as(pred))
            total += (target != -1).sum().item()

    correct = correct.item()
    test_loss = float(np.mean(losses))
    test_acc = correct / total

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total, 100. * test_acc))
        
    return test_loss, test_acc

def main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() # Add MPS for macos gpu

    # Set proper device based on cuda/mps availability 
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    print("Torch device selected: ", device)
    
    # Load datasets for training and testing
    places = datasets.Places365("./data", split="train-standard", small=True, download=True)
    chroma = torch.load("./data/chroma.pt")["train-standard"]
    train_indices, test_indices = data.select_indices(places, chroma=chroma)
    
    train_set = data.PlacesDataset(places, train_indices, FLAGS.image_size, train=True)
    test_set = data.PlacesDataset(places, test_indices, FLAGS.image_size, train=False)
    
    train_loader = DataLoader(train_set, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4, persistent_workers=True)
     
    # Load bins
    bins_data = torch.load("./data/bins.pt")
    cell_to_bin = bins_data["cell_to_bin"].to(device)
    centers = bins_data["centers"].to(device)
    weights = bins_data["weights"].to(device)
    Q = len(weights)

    # Initialize the model and send to device 
    model = ConvNet(Q).to(device)

    # Define loss function
    criterion = nn.NLLLoss(weight=weights, ignore_index=-1)
        
    # Define optimizer function.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate, momentum=0.9, weight_decay=5e-4)
        
    # Define scheduler function
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLAGS.num_epochs)
    
    best_accuracy = 0.0
    best_test_loss = math.inf
    loss_not_inc = 0
        
    # Set up data file for report analysis
    os.makedirs("data/output", exist_ok=True)
    data_file = open(f"data/output/data-{FLAGS.learning_rate}-{FLAGS.num_epochs}-{FLAGS.batch_size}.csv", mode='w', newline="", encoding="utf-8")
    writer = csv.writer(data_file)
    writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        
    # Record start time
    start_time = time.perf_counter()
        
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print(f"\nEpoch {epoch}")
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size, cell_to_bin)
        test_loss, test_accuracy = test(model, device, test_loader, criterion, cell_to_bin)
            
        scheduler.step()
            
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            
        # Save data to analysis file
        writer.writerow([epoch, train_loss, train_accuracy, test_loss, test_accuracy])
        
        if test_loss < best_test_loss:
            torch.save(model.state_dict(), f"data/output/best_model-{FLAGS.learning_rate}-{FLAGS.num_epochs}-{FLAGS.batch_size}.pt")
            best_test_loss = test_loss
            loss_not_inc = 0
        else:
            loss_not_inc += 1
            if loss_not_inc >= 8: # Quit if loss not increasing after this many
                break
            
    # Record end and total time
    end_time = time.perf_counter()
    proc_time = end_time - start_time
    td_duration = timedelta(seconds=proc_time)
        
    print("accuracy is {:2.2f}".format(100. * best_accuracy))
        
    print(f"Processing took {td_duration} total, {td_duration / FLAGS.num_epochs} per epoch")
    
    model.load_state_dict(torch.load(f"data/output/best_model-{FLAGS.learning_rate}-{FLAGS.num_epochs}-{FLAGS.batch_size}.pt"))
    visualize.save_samples(model, device, test_set, centers, f"data/output/samples-{FLAGS.learning_rate}-{FLAGS.num_epochs}-{FLAGS.batch_size}.png")

    print("Training and evaluation finished")
        
    data_file.close()



if __name__ == "__main__":
    # Set parameters for CNN Exercise
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=40,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=50,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Height/width images are resized to.')

        
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
        
    main(FLAGS)