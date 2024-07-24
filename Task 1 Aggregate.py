# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:01:53 2024

@author: austi
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import neurogym as ngym
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import MaxNLocator

# Environment
task = 'PerceptualDecisionMaking-v0'

kwargs = {'dt': 100}
seq_len = 100

# Make supervised dataset
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
env = dataset.env
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n

all_performances = np.empty(0)
all_losses = np.empty(0)
simulations = []

for s in range(5):

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


#Training----------------------------------------------------------------------

    
    s += 1
    running_loss = 0.0
    x = []
    losses = []
    performances = []
    ticks = []
    print("Simulation:", s)
    
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
            ticks.append(i + 1)
            
    #Loss during training 
    losses_graph,axes = plt.subplots()
    axes.plot(x,losses,marker = 'o', color = 'green')
    axes.set_xticks(ticks)
    axes.xaxis.set_major_formatter(EngFormatter())
    axes.set_title(("Loss Across Epochs in Simulation" + " " + str(s) + " " + "of Training"), fontsize = 18)
    axes.set_xlabel("Epochs", fontsize = 15)
    axes.set_ylabel("Loss", fontsize = 15)
    
    #Performance of task during training
    performances_graph,axes = plt.subplots()
    axes.plot(x,performances,marker = 'o', color = 'green')
    axes.set_xticks(ticks)
    axes.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
    axes.xaxis.set_major_formatter(EngFormatter())
    axes.set_title("Performance Across Epochs in Simulation" + " " + str(s) + " " + "of Training", fontsize = 18)
    axes.set_xlabel("Epochs", fontsize = 15)
    axes.set_ylabel("Performance", fontsize = 15)
    
    if s == 1:
        performances_graph.savefig('C:/Users/austi/Downloads/Project/Graphs/Performance of Task 1 (Simulation 1).svg')
    
    print('Finished Training')
    print('')
#------------------------------------------------------------------------------
    
#Testing-----------------------------------------------------------------------
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
    print('Average performance in {:d} trials of simulation {:d}:'.format(num_trial, s))
    print(perf)
    print('')
    all_performances = np.append(all_performances, perf)
    simulations.append(s)

#------------------------------------------------------------------------------ 

#Analysis

#Calculates and prints the average of the performance across all the simulations.

average_performance = np.mean(all_performances,0)
print("Average performance across all trials:", average_performance)
print('')

#Graphs the performance of each simulation.
all_performances_graph,axes = plt.subplots()
axes.plot(simulations,all_performances,marker = 'o', color = 'green')
axes.axhline(y=average_performance, linestyle = 'dotted', color = 'green')
axes.xaxis.set_major_locator(MaxNLocator(integer=True))
axes.set_xticks(simulations)
axes.set_title("Performances of Task 1 Per Simulation of Training", fontsize = 18)
axes.set_xlabel("Simulation #", fontsize = 15)
axes.set_ylabel("Performance", fontsize = 15)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
axes.legend()
all_performances_graph.savefig('C:/Users/austi/Downloads/Project/Graphs/Performances of Task 1.svg')