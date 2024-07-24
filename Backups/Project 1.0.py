# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:26:23 2024

@author: austi
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import neurogym as ngym

# Environment
task = 'PerceptualDecisionMaking-v0'
kwargs = {'dt': 100}
seq_len = 100

# Creation of first dataset and layers
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
env = dataset.env
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n

class Net(nn.Module):
    def __init__(self, num_h):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(ob_size, num_h,num_layers=1)
        self.linear = nn.Linear(num_h, act_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        x = self.linear(out)
        return x
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net(num_h=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

#Training #1----------------------------------------------------------------------

running_loss = 0.0
x = []
losses = []

print("Task 1 training:")

for i in range(2000):
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)

    loss = criterion(outputs.view(-1, act_size), labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 200 == 199:
        print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
        x.append(i + 1)
        losses.append(running_loss / 200)
        running_loss = 0.0

loss_task1,axes = plt.subplots()
axes.plot(x,losses,marker = 'o')
axes.set_xticks(x)
axes.set_title("Loss Across Epochs During Task 1 Training")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss")
print('1st round of training complete')
print('')

#Training 2----------------------------------------------------------------------

task2 = 'PerceptualDecisionMakingDelayResponse-v0'
dataset2 = ngym.Dataset(task2, env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
x = []
losses = []
print("Task 2 training:")

for i in range(2000):
    inputs, labels = dataset2()
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)

    loss = criterion(outputs.view(-1, act_size), labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 200 == 199:
        print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
        x.append(i + 1)
        losses.append(running_loss / 200)
        running_loss = 0.0

loss_task2,axes = plt.subplots()
axes.plot(x,losses,marker = 'o')
axes.set_xticks(x)
axes.set_title("Loss Across Epochs During Task 2 Training")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss")
print('1st round of training complete')
print('')
print('2nd round of training complete')
print('')

#Test 1-------------------------------------------------------------------------

perf = 0
num_trial = 200
for i in range(num_trial):
    env.new_trial()
    ob, gt = env.ob, env.gt
    ob = ob[:, np.newaxis, :]  # Add batch axis
    inputs = torch.from_numpy(ob).type(torch.float).to(device)

    action_pred = net(inputs)
    action_pred = action_pred.cpu().detach().numpy()
    action_pred = np.argmax(action_pred, axis=-1)
    perf += gt[-1] == action_pred[-1, 0]

perf /= num_trial
print('Average performance on task 1 ("PerceptualDecisionMaking-v0") in {:d} trials:'.format(num_trial))
print(perf)

#Test 2-------------------------------------------------------------------------

#Change the environment to task 2 ("PerceptualDecisionMakingDelayResponse-v0")
env = dataset2.env

perf = 0
num_trial = 200
for i in range(num_trial):
    env.new_trial()
    ob, gt = env.ob, env.gt
    ob = ob[:, np.newaxis, :]  # Add batch axis
    inputs = torch.from_numpy(ob).type(torch.float).to(device)

    action_pred = net(inputs)
    action_pred = action_pred.cpu().detach().numpy()
    action_pred = np.argmax(action_pred, axis=-1)
    perf += gt[-1] == action_pred[-1, 0]

perf /= num_trial
print('Average performance on task 2 ("PerceptualDecisionMakingDelayResponse-v0") in {:d} trials:'.format(num_trial))
print(perf)
