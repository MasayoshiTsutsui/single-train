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
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from model import Conv6
DEBUG = 1

def dbg_print(s):
    if DEBUG:
        print(s)



from collections import OrderedDict

def train(model, device, train_loader, criterion, epochs):
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad])
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

def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
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
        test_dataset[0], batch_size=50, shuffle=False, **kwargs)
    print(train_dataset)
    print(test_dataset[0])
    print(test_dataset[1])
    exit(1)
    
    model = Conv6()

    model = model.to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    criterion = nn.CrossEntropyLoss()
    for epoch in range(250):
        train(model, device, train_loader, criterion, 1)
        loss, accuracy = test(model, device, criterion, test_loader)
        print(f"{epoch}, {loss}, {accuracy}")

if __name__ == '__main__':
    main()