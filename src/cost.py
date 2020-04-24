import torch
import torch.nn as nn
from utils import Data
import matplotlib.pyplot as plt

dataset = Data(switchInputAndOutput = True)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [5500, 827])

batch_size = 8
learning_rate = 0.000001
num_epochs = 80000
rho=0.6
eps=5e-05
weight_decay=0.005
bias = torch.tensor([
    10,
    10,
    1,
    20,
    10,
    5,
    5
])

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
        self.hid1 = nn.Linear(dataset.inputCount, 120)
        self.hid1_a = nn.LeakyReLU(0.1)
        self.hid1_d = nn.Dropout(p=0.02)
        self.hid2 = nn.Linear(120, 90)
        self.hid2_a = nn.LeakyReLU(0.2)
        self.hid2_d = nn.Dropout(p=0.04)
        self.output = nn.Linear(90, dataset.outputCount)
        self.output_a = nn.LeakyReLU()

    def forward(self, x):
        '''
        Exercise - Forward Propagate through the layers as defined above. Fill in params in place of ...
        '''
        x = self.hid1_d(
            self.hid1_a(
            self.hid1(x)
            )
            )
        x = self.hid2_d(self.hid2_a(self.hid2(x)))
        x = self.output_a(self.output(x))
        return x
if __name__ == '__main__':
    model = NeuralNet().to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=rho, eps=eps, weight_decay=weight_decay)

    costs = []
    errors = []

    def calculate_loss(output, target):
        loss = (output - target)**2
        loss = torch.mul(loss, bias)
        return torch.mean(loss)

    minError = 1000000000
    minEpoch = -1

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        cost = 0
        for i, (input, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            input = input.to(device)
            labels = labels.to(device)

            # Forward pass

            '''Exercise - Get Network outputs with forward propagation with current network weights'''
            outputs = model(input)
            '''Exercise - Get Loss by comparing outputs with True Labels after forward propagation'''
            outputs = outputs + torch.mul(labels - outputs, bias)
            loss = criterion(outputs, labels)

            cost += loss.item()

            # Backward and optimize

            '''Exercise - ... below needs to be replaced with functions'''

            optimizer.zero_grad() #'''Exercise - clear the gradients after each pass - Strongly recommended'''
            loss.backward() #'''Backpropagate the Loss to calculate gradient for each weight'''
            optimizer.step() #'''Update the weight using the learning rate'''
        print("epoch: {} cost: {}".format(epoch, cost))
        costs.append(cost)

        with torch.no_grad(): # In test phase, we don't need to compute gradients (for memory efficiency)
                totalError = 0
                for input, labels in test_loader:
                    '''Exercise - Move input to device after appropriate reshaping'''
                    input = input.to(device)

                    '''Exercise - Move labels to device'''
                    labels = labels.to(device)

                    #get network outputs
                    outputs = model(input)
                    error = abs(labels - outputs.data)
                    totalError += error.sum().item()
                if totalError < minError:
                    minError = totalError
                    minEpoch = epoch
                errors.append(totalError)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), 'saves/cost_e_{}_c_{}_ep_{}.model'.format(str(errors[-1]), str(costs[-1]), str(epoch)))
    torch.save(model.state_dict(), 'saves/cost_e_{}_c_{}_finish.model'.format(str(errors[-1]), str(costs[-1]), str(epoch)))
    print("ending error {}".format(errors[-1]))
    print("best error of {} at epoch {}".format(minError, minEpoch))
    plt.plot(costs, label="cost", color="blue")
    plt.legend()
    plt.show()

    plt.plot(errors, label="error", color="red")
    plt.legend()
    plt.show()
