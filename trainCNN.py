## trainCNN.py is the cilent of the training and evalating process of the CNN. The model used Lenet5 structure.


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import time
import copy
import os.path as osp

from utils import Config
from LeNet5 import nets
from pipeline2 import get_dataloader

import numpy as np
import matplotlib.pyplot as plt





## the training process function for one iteration
## input: training batch dataloader; CNN model net; loss function criterion; optimizer; GPU device
## the CNN model nets after one iteration

def training(dataloader, nets, criterion, optimizer, device):
    for data in dataloader:
        images,ylabels = data
        #print(images.shape)
        images = images.to(device)
        ylabels = ylabels.to(device)
        #ylabels = ylabels.view(-1,1) 
        optimizer.zero_grad()
        outputs = nets(images)
        batch_loss = criterion(outputs, ylabels)
        batch_loss.backward()
        optimizer.step()
    return nets

## the test process function 
## input: evaluation batch dataloader; CNN model nets; GPU device; train/test flag phase
## output: evaluation accuracy
def testing(dataloader, nets, device, phase):
    corrects = 0
    total_corrects = 0
    for data in dataloader:
        images,ylabels = data
        images = images.to(device)
        ylabels = ylabels.to(device) 
        #ylabels = ylabels.view(-1,1)
        with torch.no_grad():
            outputs = nets(images)
            _, pred = torch.max(outputs.data, 1)
            corrects += (pred==ylabels).sum().item()
            total_corrects+= ylabels.size(0)
        
    accuracy = corrects / total_corrects
    if phase == 'train':
        print('Training accuracy: ',accuracy)
    else:
        print('Testing accuracy: ',accuracy)
    return accuracy





if __name__=='__main__':
    dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    trainloader = dataloaders['train']
    testloader = dataloaders['test']
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nets.parameters(), lr= Config['learning_rate'])
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    nets.to(device)
    best_accuracy = 0.0
    train_accuracy = np.zeros(Config['num_epochs'])
    test_accuracy = np.zeros(Config['num_epochs'])


    ## training for number of epochs
    for epoch in range(Config['num_epochs']):
        print('Epoch ',epoch, ':')
        nets = training(trainloader, nets, criterion,optimizer,device)
        accuracy = testing(trainloader, nets, device, 'train')
        train_accuracy[epoch] = accuracy    
        accuracy = testing(testloader, nets, device, 'test')
        test_accuracy[epoch] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    

    print('Best test accuracy: ',best_accuracy)



    ## plot the learning curves
    plt.figure()
    plt.plot(train_accuracy,'r',label = 'train_accuracy')
    plt.plot(test_accuracy,'b',label = 'test_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Learning Curves')
    plt.savefig('learning_acc.png', dpi=256)
