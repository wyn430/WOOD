import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(24)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


from Model import FashionCNN, DenseNet3
from Dataset import Fashion_MNIST, MNIST, Cifar_10, SVHN, TinyImagenet_r, TinyImagenet_c
from WOOD_Loss import NLLWOOD_Loss_v2, sink_dist_test_v2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##parameters, Fashion MNIST is in distribution dataset, and MNIST is out of distribution dataset


beta = torch.Tensor([float(sys.argv[1])]).to(device)
num_epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
InD_batch_size = int(sys.argv[4])
InD_Dataset = str(sys.argv[5])
OOD_Dataset = str(sys.argv[6])
C = int(sys.argv[7])

OOD_batch_size = batch_size - InD_batch_size
test_batch_size = 100
learning_rate = 0.001
##parameters in loss
num_class = torch.LongTensor([10]).to(device)

data_dic = {
    'MNIST': MNIST,
    'FashionMNIST': Fashion_MNIST, 
    'Cifar10': Cifar_10,
    'SVHN': SVHN, 
    'Imagenet_r': TinyImagenet_r,
    'Imagenet_c': TinyImagenet_c
}

InD_train_loader, InD_test_loader = data_dic[InD_Dataset](InD_batch_size, test_batch_size)
OOD_train_loader, OOD_test_loader = data_dic[OOD_Dataset](OOD_batch_size, test_batch_size)


##load model
model = DenseNet3(depth=100, num_classes=10, input_channel = C)
model.to(device)
model = nn.DataParallel(model)
print("Let's use", torch.cuda.device_count(), "GPUs!")



##load loss function
NLLWOOD_l = NLLWOOD_Loss_v2.apply
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

file_root = './runs/' + time.strftime("%Y%m%d-%H%M%S") + '/'
os.mkdir(file_root)
file_name = file_root + 'log.txt'
best_test_acc = 0
best_OOD_dist = 1
best_then01_dist = 1

with open(file_name, 'a') as f:
    f.write('DenseNet 100 f ' + InD_Dataset + ' InD ' + OOD_Dataset + ' OOD experiment epoch = ' + str(num_epochs) + ' beta = ' + str(beta[0]) + ' OOD Size = ' + str(OOD_batch_size) + '\n')


for epoch in range(num_epochs):
    count = 0
    for (InD_images, InD_labels), (OOD_images, OOD_labels) in zip(InD_train_loader, OOD_train_loader):
    #for InD_images, InD_labels in InD_train_loader:
        model.train()
        
        ##load a batch of ood data
        ##change the label of ood data
        OOD_labels[:] = num_class[0]

        images = torch.cat([InD_images, OOD_images], dim=0)
        labels = torch.cat([InD_labels, OOD_labels], dim=0)

        ##shuffle the order of InD and OOD samples
        idx = torch.randperm(images.shape[0])
        images = images[idx].view(images.size())
        labels = labels[idx].view(labels.size())

        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)

        train = Variable(images)
        #print(train.shape)
        labels = Variable(labels)

        # Forward pass 
        outputs = model(train)
        
        loss = NLLWOOD_l(outputs, labels, num_class, beta, device)

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        #Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1
        
        
        # Testing the model
        
        if not (count % 400):    # It's same as "if count % 100 == 0"
            total = 0
            correct = 0
            InD_test_sink_dist_list = []
            model.eval()
            ##InD samples test
            for images, labels in InD_test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)

                test = Variable(images.view(images.size()))

                outputs = model(test)


                InD_sink_dist = sink_dist_test_v2(outputs, labels, num_class[0], device).cpu().detach().numpy()
                InD_test_sink_dist_list.append(InD_sink_dist)


                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)

                correct += (predictions == labels).sum()

                total += len(labels)
            
            InD_test_mean_sink_dist = np.concatenate(InD_test_sink_dist_list, axis=0)
            InD_sink_mean = np.mean(InD_test_mean_sink_dist)
            
            accuracy = correct * 100 / float(total)

            if accuracy >= best_test_acc:
                best_test_acc = accuracy
                torch.save(model.state_dict(), '%s/acc_model.t7' % file_root)

            ##OOD samples test
            OOD_test_sink_dist_list = []
            total = 0
            for images, labels in OOD_test_loader:
                labels[:] = 0
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)

                test = Variable(images.view(images.size()))

                outputs = model(test)
                total += len(labels)
                OOD_sink_dist = sink_dist_test_v2(outputs, labels, num_class[0], device).cpu().detach().numpy()
                OOD_test_sink_dist_list.append(OOD_sink_dist)
            
            OOD_test_mean_sink_dist = np.concatenate(OOD_test_sink_dist_list, axis=0)
            OOD_sink_mean = np.mean(OOD_test_mean_sink_dist)
            thresh = np.quantile(InD_test_mean_sink_dist, 0.95)
            fpr = OOD_test_mean_sink_dist[OOD_test_mean_sink_dist<=thresh].shape[0] / float(OOD_test_mean_sink_dist.shape[0])
            thresh = float(format(thresh, '.4g'))
            fpr = float(format(fpr, '.4g'))
            accuracy = float(format(accuracy, '.4g'))
            
            log = "Epoch: {}, Iteration: {}, Loss: {}, Accuracy: {}%, InD_sink: {}, InD_95_thresh: {}, OOD_sink: {}, OOD_95_FPR: {}".format(epoch, count, loss[0], accuracy, InD_sink_mean, thresh, OOD_sink_mean, fpr)
            print(log)

            with open(file_name, 'a') as f:
                f.write(log+'\n')
            if fpr <= best_OOD_dist:
                
                best_OOD_dist = fpr
                torch.save(model.state_dict(), '%s/OOD_model_%s_%s.t7' % (file_root, str(accuracy).split('.')[0], 
                                                                          str(fpr).split('.')[1]))
            
            






