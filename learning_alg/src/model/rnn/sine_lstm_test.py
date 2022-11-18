#! /usr/bin/env python3

import torch
from torch import nn
import matplotlib.pyplot as plt
import random
import numpy as np

interval = 0.01
batch_size = 1
time_step = 10
n_direction = 1
n_layers = 1

def sine(n, e = 0):
    '''
    Produce a series of discrete sine values.
    @param n: the length of the produced series.
    '''
    return torch.sin(torch.arange(e, e + n * interval, interval))

def data_iter(data, time_step):
    '''
    Iterator for producing traning data or test data.
    @param data: raw series data.
    @param time_step: time step input to LSTM.
    '''
    sample_indices = list(range(len(data) - time_step - 1))
    random.shuffle(sample_indices)
    for i in sample_indices:
        yield data[i: i + time_step], data[i + 1: i + time_step + 1]

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.state = self.init_state()
    
    def init_state(self):
        return (
            torch.zeros(n_direction * n_layers, batch_size, self.hidden_size),
            torch.zeros(n_direction * n_layers, batch_size, self.hidden_size)
        )
    
    def reset_state(self):
        self.state = self.init_state()
    
    def forward(self, X):
        out, self.state = self.lstm(torch.tensor(X).view(len(X), batch_size, -1), self.state)
        out = self.linear(out.view(len(out), -1))
        return out

data = sine(10000)
n_epochs = 1
input_size = 1
hidden_size = 10
model = LSTMNet(input_size, hidden_size)
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
loss_fn = torch.nn.MSELoss()

losses = []
for i in range(n_epochs):
    err = []
    for train, label in data_iter(data, time_step):
        optimizer.zero_grad()
        model.reset_state()
        y_hat = model(train)
        loss = loss_fn(y_hat, label)
        err.append(loss.item())
        loss.backward()
        optimizer.step()
    err = np.mean(err)
    print('The loss of %dth epoch is %.2f' % (i, err))
    losses.append(err)

test = sine(10000, 2)
preds = []
for i in test:
    preds.append(model([i]).item()) 

plt.plot([i for i in range(len(test))], test, 'r')
plt.plot([i for i in range(len(test))], preds, 'g')
plt.show()
