# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:07:09 2024

@author: austi
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import neurogym as ngym
from matplotlib.ticker import EngFormatter

# Environment
task = 'PerceptualDecisionMaking-v0'
task2 = 'PerceptualDecisionMakingDelayResponse-v0'

kwargs = {'dt': 100}
seq_len = 100

# Make supervised dataset
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

#training----------------------------------------------------------------------
running_loss = 0.0
x = []
losses = []
performances = []
ticks = []

for i in range(4000):
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
        
        x.append(i + 1)
        losses.append(running_loss / 200)
        num_trial = 200

        perf = 0
        for j in range(num_trial):
            
            env.new_trial()
            ob, gt = env.ob, env.gt
            ob = ob[:, np.newaxis, :]  # Add batch axis
            inputs = torch.from_numpy(ob).type(torch.float).to(device)

            action_pred = net(inputs)
            action_pred = action_pred.cpu().detach().numpy()
            action_pred = np.argmax(action_pred, axis=-1)
            perf += gt[-1] == action_pred[-1, 0]

        perf /= num_trial
        performances.append(perf)
        
        print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
        running_loss = 0.0
    
    if i % 1000 == 999:
        ticks.append(i+1)
        
#Loss during training 
losses_graph,axes = plt.subplots()
axes.plot(x,losses,marker = 'o', color = 'green')
axes.set_xticks(ticks)
axes.xaxis.set_major_formatter(EngFormatter())
axes.set_title("Loss Across Epochs", fontsize = 18)
axes.set_xlabel("Epochs", fontsize = 15)
axes.set_ylabel("Loss", fontsize = 15)

#Performance of task during training
performances_graph,axes = plt.subplots()
axes.plot(x,performances,marker = 'o', color = 'green')
axes.axhline(y = 0.5, color = 'black', linestyle = 'dashed')
axes.set_xticks(ticks)
axes.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
axes.xaxis.set_major_formatter(EngFormatter())
axes.set_title("Performance on Task 1 Throughout Training", fontsize = 18)
axes.set_xlabel("Epochs", fontsize = 15)
axes.set_ylabel("Performance", fontsize = 15)
axes.legend(['Task 1','Chance'], loc = 'lower right', prop = {'size':13})
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
performances_graph.savefig('C:/Users/austi/Downloads/Project/Graphs/Performance on Task 1 Throughout Training.svg')

print('Finished Training')
print('')
#------------------------------------------------------------------------------

#testing-----------------------------------------------------------------------
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
print('Average performance in {:d} trials'.format(num_trial))
print(perf)
#------------------------------------------------------------------------------ 