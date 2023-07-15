# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:28:21 2023

@author: Paras
"""
import torch
from torch import nn
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import wandb

from ViT import VisionTransformer

wandb.login(key='434d12235bff28857fbf238c1278bdacead1838d')
wandb.init(project='vit_distributed',name='mnist_10_2')

batch_size = 16
# Define the transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# Load the training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#torch.autograd.set_detect_anomaly(True)
model = VisionTransformer(batch_size=batch_size,
                          num_classes=10,
                          image_size=28,
                          patch_size=4,
                          in_channels=1,
                          embed_dim=512,
                          num_heads=8,
                          num_layers=2,
                          dim_feedforward=2048,
                          dropout=0.1)

'''
num_classes,
image_size,
patch_size,
in_channels,
embed_dim,
num_heads,
num_layers,
dim_feedforward=2048,  # default
dropout=0.1,           # default
activation="relu",     # default (or use gelu)
norm_first=False)
'''

model = model.to(device)
print(model)
num_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:',num_params)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=3e-4)

epoch_losses = []
epoch_accuracies = []
prev_loss = float('inf')

for epoch in range(25):  # Number of training epochs

    epoch_loss = []
    epoch_acc = []
    model.train()
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        c = model(images)
        loss = criterion(c,labels)

        with torch.no_grad():
            predictions = torch.argmax(c, dim=-1)

        acc = torch.sum(predictions == labels)/batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        epoch_acc.append(acc.cpu().numpy())
        #print('Batch',i,'trained!')

    model.eval()
    epoch_losses.append(np.average(epoch_loss))
    epoch_accuracies.append(np.average(epoch_acc))
    wandb.log({'CrossEntropyLoss':epoch_losses[-1],'Accuracy':epoch_accuracies[-1]})
    print('Epoch',epoch,'loss:',epoch_losses[-1])
    print('Epoch',epoch,'accuracy:',epoch_accuracies[-1])
    if epoch > 0 and abs(epoch_losses[-1] - prev_loss) < 0.003:
        print('Training stopped. Loss difference threshold reached.')
        break

    prev_loss = epoch_losses[-1]

test_losses = []
test_accuracies = []
model.eval()

for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        output = model(images)
        loss = criterion(output, labels)
        predictions = torch.argmax(output, dim=-1)
        acc = torch.sum(predictions == labels)/batch_size

    test_losses.append(loss.item())
    test_accuracies.append(acc.cpu().numpy())

average_test_loss = np.average(test_losses)
average_test_accuracy = np.average(test_accuracies)

wandb.log({'Test Loss': test_losses, 'Test Accuracy': test_accuracies})


print('Test Loss:', average_test_loss)
print('Test Accuracy:', average_test_accuracy)

wandb.finish()
'''
def __init__(self,
             num_classes,
             image_size,
             patch_size,
             in_channels,
             embed_dim,
             num_heads,
             num_layers,
             dim_feedforward=2048,  # default
             dropout=0.1,           # default
             activation="relu",     # default (or use gelu)
             norm_first=False):
'''
