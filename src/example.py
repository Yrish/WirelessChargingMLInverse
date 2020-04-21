import torch
import torch.nn as nn
from utils import Data
import numpy as np

dataset = Data()
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [2000, 812])

batch_size = 4
learning_rate = 0.002
num_epochs = 10

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        '''
        Exercise - Define the N/w architecture. Use RELU Activation
        '''
        self.hid1 = nn.Linear(dataset.inputCount, 90)
        self.hid1_a = nn.LeakyReLU(0.1)
        self.hid1_d = nn.Dropout(p=0.02)
        self.output = nn.Linear(90, dataset.outputCount)
        self.output_a = nn.LeakyReLU()

    def forward(self, x):
        '''
        Exercise - Forward Propagate through the layers as defined above. Fill in params in place of ...
        '''
        x = self.hid1_d(self.hid1_a(self.hid1(x)))
        x = self.output_a(self.output(x))
        return x

#model = NeuralNet().to(device)

#from invertible_resnet.models import model_utils
import sys
import os, os.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'invertible_resnet'))

from invertible_resnet.models import conv_iResNet
#model = model_utils.ActNorm(7).to(device)
input_shape = (7,1,1,)
model = conv_iResNet.conv_iresnet_block(input_shape, 9)

criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    print("Beginning epoch {0:d}/{1:d}".format(epoch + 1, num_epochs))
    for i, (input, labels) in enumerate(train_loader):
        if i % 100 == 0:
            print(" - i: {0:d}/{1:d}".format(i, len(train_loader)))
        # Move tensors to the configured device
        input = input.to(device)
        labels = labels.to(device)

        # Forward pass

        '''Exercise - Get Network outputs with forward propagation with current network weights'''
        outputs = model(input)
        '''Exercise - Get Loss by comparing outputs with True Labels after forward propagation'''
        loss = criterion(outputs, labels)

        # Backward and optimize

        '''Exercise - ... below needs to be replaced with functions'''

        optimizer.zero_grad() #'''Exercise - clear the gradients after each pass - Strongly recommended'''
        loss.backward() #'''Backpropagate the Loss to calculate gradient for each weight'''
        optimizer.step() #'''Update the weight using the learning rate'''

with torch.no_grad(): # In test phase, we don't need to compute gradients (for memory efficiency)
        totalError = 0
        # all_outputs:
        #   2-D array.  Rows are items.  Columns within a row are one of the 7
        #   outputs.
        # all_error:
        #   same shape, but output (i.e. predictions) - labels (i.e. expected)
        shape = (len(test_loader) * len(train_dataset), dataset.outputCount)
        all_outputs, all_labels, all_errors = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        #for input, labels in test_loader:
        for i, (input, labels) in enumerate(test_loader):
            '''Exercise - Move input to device after appropriate reshaping'''
            input = input.to(device)

            '''Exercise - Move labels to device'''
            labels = labels.to(device)

            #get network outputs
            outputs = model(input)
            _, predicted = torch.max(outputs.data, 1)
            error = abs(labels - outputs.data)
            totalError += error.sum().item()

            labels = labels.to('cpu')
            end = i + len(labels)
            all_outputs[i:end] = outputs
            all_labels [i:end] = labels
            all_errors [i:end] = labels - outputs

        print('Error of the network on the test input: {}'.format(error.sum()))

        for i in range(dataset.outputCount):
            predictions = all_outputs[:][i]
            labels      = all_labels [:][i]
            errors      = all_errors [:][i]
            squared_errors = np.square(errors)
            absolute_errors = np.abs(errors)
            print("")
            print("NN output #{0:d}:".format(i+1))
            print("  - predictions mean: {0:f}".format(predictions.mean()))
            print("  - predictions stddev: {0:f}".format(predictions.std()))
            print("  - labels mean: {0:f}".format(labels.mean()))
            print("  - labels stddev: {0:f}".format(labels.std()))
            print("  - errors mean: {0:f}".format(errors.mean()))
            print("  - errors stddev: {0:f}".format(errors.std()))
            print("  - squared errors mean: {0:f}".format(squared_errors.mean()))
            print("  - squared errors stddev: {0:f}".format(squared_errors.std()))
            print("  - absolute errors mean: {0:f}".format(absolute_errors.mean()))
            print("  - absolute errors stddev: {0:f}".format(absolute_errors.std()))
