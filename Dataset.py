import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import os

def Fashion_MNIST(batch_size, test_batch_size):
    
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size)
    
    return train_loader, test_loader


def MNIST(batch_size, test_batch_size):
    
    train_set = torchvision.datasets.MNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.MNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size)
    
    return train_loader, test_loader


def Cifar_10(batch_size, test_batch_size):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         #std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data/cifar10', 
                                                                train=True,download=True,
                                                                transform=transform),
                                               batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./datasets/cifar10', train=False,download=True, 
                                                              transform=transform),
                                             batch_size=test_batch_size, shuffle=True)
    
    return train_loader, val_loader


def SVHN(batch_size, test_batch_size):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_loader = torch.utils.data.DataLoader(datasets.SVHN('./data/svhn/', 
                                                         split='train',
                                                         transform=transform,
                                                         download=True),
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(datasets.SVHN('./data/svhn/', split='test',
                                                       transform=transform, 
                                                       download=True),
                                             batch_size=test_batch_size, shuffle=True)
    
    return train_loader, val_loader

def TinyImagenet_r(batch_size, test_batch_size):
    transform = transforms.Compose([transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_datasets = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'train'), transform=transform) 
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    
    test_datasets = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'test'), transform=transform) 
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=test_batch_size, shuffle=True)
    
    return train_loader, test_loader


def TinyImagenet_c(batch_size, test_batch_size):
    transform = transforms.Compose([transforms.RandomCrop(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_datasets = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'train'), transform=transform) 
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    
    test_datasets = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'test'), transform=transform) 
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=test_batch_size, shuffle=True)
    
    return train_loader, test_loader
    
    
    
    
    