# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import os
import datetime
import numpy as np
import random
from numpy.core.fromnumeric import sort

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from model import ResNet18
DEBUG = 1

def dbg_print(s):
    if DEBUG:
        print(s)



from collections import OrderedDict

def train(model, device, train_loader, criterion, epochs, test_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.05,
                                        momentum=0.9, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    model.train()
    for epoch in range(1, epochs+1):
        runningloss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            runningloss += loss.item()
        runningloss /= len(train_loader)
        loss, accuracy = test(model, device, criterion, test_loader)
        scheduler.step()
        print(f"{epoch}, {runningloss}, {loss}, {accuracy}")

def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    test_loss /= len(test_loader)
    acc = correct / len(test_loader.dataset)
    return test_loss, acc

def main():



    device = torch.device("cuda")

    kwargs = {'num_workers': 4, 'pin_memory': False}

    train_dataset = datasets.CIFAR10(os.path.join('../data', 'cifar10'), train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ]))
    test_dataset = datasets.CIFAR10(os.path.join('../data', 'cifar10'), train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])),

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset[0], batch_size=100, shuffle=False, **kwargs)
    
    model = ResNet18()

    model = model.to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    criterion = nn.CrossEntropyLoss()
    train(model, device, train_loader, criterion, 30, test_loader)

if __name__ == '__main__':
    main()