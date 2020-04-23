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


# constraints:
# Iin[A]: 1-100
# Iout[A]: 1-100
# win [mm]: 1-1000
# p1[mm]: 3-150
# p2[mm]: 3-150
# p3[mm]: 3-150
# l[mm]: 3-1500
#
# V_PriWInd[cm3]: 1-300
# Vcore[cm3]: 1-10000
# kdiff[%]: 0.1-40
# Pout[W]: 1-3000
# Bleak[uT]: 1-300

# From 7 sim input channels to 5 sim output channels
class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        '''
        Exercise - Define the N/w architecture. Use RELU Activation
        '''
        self.net = nn.Sequential(
            nn.Linear(dataset.inputCount, 90),  # hid1
            nn.LeakyReLU(0.1),  # hid1_a
            nn.Dropout(p=0.02),  # hid1_d
            nn.Linear(90, dataset.outputCount),  # output
            nn.LeakyReLU(),  # output_a
            ## 7 simulation inputs: Iin[A}, win[mm], p1[mm], p2[mm], p3[mm], l[mm]
            #nn.Hardtanh(min_val=[1,1,1,3,3,3,3,],max_val=[100,100,1000,150,150,150,1500,]),
            ## 5 simulation outputs: V_PriWInd[cm3], Vcore[cm3], kdiff[%], Pout[W], Bleak[uT]
            ##nn.Hardtanh(min_val=[1,1,0.1,1,1],max_val=[300,10000,40,3000,300,]),
        )

    def forward(self, x):
        '''
        Exercise - Forward Propagate through the layers as defined above. Fill in params in place of ...
        '''
        #x = self.hid1_d(self.hid1_a(self.hid1(x)))
        #x = self.output_a(self.output(x))
        x = self.net(x)
        return x

model = NeuralNet().to(device)

##from invertible_resnet.models import model_utils
#import sys
#import os, os.path
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'invertible_resnet'))
#
#from invertible_resnet.models import conv_iResNet
##model = model_utils.ActNorm(7).to(device)
#input_shape = (7,1,1,)
#model = conv_iResNet.conv_iresnet_block(input_shape, 9)

#import sys
#import os, os.path
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memcnn'))
#import memcnn  # memcnn/memcnn
#from memcnn.models.resnet import ResNet, RevBasicBlock
#block = RevBasicBlock(7, 7)
#model = ResNet(block=block, layers=4, num_classes=5, channels_per_layer=[7,8,8,5])










if False:
    import sys
    import os, os.path
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memcnn'))

    import torch
    import torch.nn as nn
    #import memcnn

    import memcnn  # memcnn/memcnn
    from memcnn.models.resnet import ResNet, RevBasicBlock


    # turn the ExampleOperation invertible using an additive coupling
    invertible_module = memcnn.AdditiveCoupling(
        Fm=NeuralNet().to(device),
        Gm=NeuralNet().to(device),
    )

    # test that it is actually a valid invertible module (has a valid inverse method)
    #assert memcnn.is_invertible_module(invertible_module, test_input_shape=(100,7,))

    # wrap our invertible_module using the InvertibleModuleWrapper and benefit from memory savings during training
    invertible_module_wrapper = memcnn.InvertibleModuleWrapper(fn=invertible_module, keep_input=True, keep_input_inverse=True)

    ## by default the module is set to training, the following sets this to evaluation
    ## note that this is required to pass input tensors to the model with requires_grad=False (inference only)
    #invertible_module_wrapper.eval()
    #
    ## test that the wrapped module is also a valid invertible module
    #assert memcnn.is_invertible_module(invertible_module_wrapper, test_input_shape=X.shape)
    #
    ## compute the forward pass using the wrapper
    #Y2 = invertible_module_wrapper.forward(X)
    #
    ## the input (X) can be approximated (X2) by applying the inverse method of the wrapper on Y2
    #X2 = invertible_module_wrapper.inverse(Y2)
    #
    ## test that the input and approximation are similar
    #assert torch.allclose(X, X2, atol=1e-06)
    model = invertible_module_wrapper












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
