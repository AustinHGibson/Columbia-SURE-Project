# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:52:03 2024

@author: austi
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import neurogym as ngym
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import MaxNLocator

# Defines the environment.
task = 'PerceptualDecisionMaking-v0'
task2 = 'PerceptualDecisionMakingDelayResponse-v0'
num_trial = 200

kwargs = {'dt': 100}
seq_len = 100

# Creation of datasets and layers of the network
dataset1 = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
dataset2 = ngym.Dataset(task2, env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)
env1 = dataset1.env
env2 = dataset2.env

ob_size = env1.observation_space.shape[0]
act_size = env1.action_space.n

all_performances_task1 = []
all_performances_task2 = []

for s in range(10):
    
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
    ticks = [1000,2000,3000,4000]
    losses = []
    performances_task1 = []
    performances_task2 = []

    print("Simulation:", s)
    print("Task 1 training:")
    
    #Task 2 Training
    for i in range(2000):
        inputs, labels = dataset2()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
    
        # Reset the parameter gradients.
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs.view(-1, act_size), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    #At intervals of 200 epochs, calculate the loss and performance and print them.
        if i % 200 == 199:
            
            x.append(i + 1)
            losses.append(running_loss / 200)
            
            #Performance test on Task 1
            perf = 0
            for j in range(num_trial):
                
                env1.new_trial()
                ob, gt = env1.ob, env1.gt
                ob = ob[:, np.newaxis, :]  # Add batch axis
                inputs = torch.from_numpy(ob).type(torch.float).to(device)
    
                action_pred = net(inputs)
                action_pred = action_pred.cpu().detach().numpy()
                action_pred = np.argmax(action_pred, axis=-1)
                perf += gt[-1] == action_pred[-1, 0]
    
            perf /= num_trial
            performances_task1.append(perf)
            
            print('{:d} loss: {:0.5f}   Performance(T1): {:0.5f}'.format(i + 1, running_loss / 200, perf))
            running_loss = 0.0   
            
            #Performance test on Task 2
            perf = 0
            for j in range(num_trial):
                
                env2.new_trial()
                ob, gt = env2.ob, env2.gt
                ob = ob[:, np.newaxis, :]  # Add batch axis
                inputs = torch.from_numpy(ob).type(torch.float).to(device)
    
                action_pred = net(inputs)
                action_pred = action_pred.cpu().detach().numpy()
                action_pred = np.argmax(action_pred, axis=-1)
                perf += gt[-1] == action_pred[-1, 0]
    
            perf /= num_trial
            performances_task2.append(perf)
            
            print('                    Performance(T2): {:0.5f}'.format(perf))
            print('')
            running_loss = 0.0                
            
#------------------------------------------------------------------------------            
        
    #Task 1 Training
    for i in range(2000):
        inputs, labels = dataset1()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
    
        #Reset the parameter gradients.
        optimizer.zero_grad()
    
        #forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs.view(-1, act_size), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    #At intervals of 200 epochs, calculate the loss and performance and print them.    
        if i % 200 == 199:
            
            x.append(i + 2001)
            losses.append(running_loss / 200)
    
            #Performance test on Task 1
            perf = 0
            for j in range(num_trial):
                
                env1.new_trial()
                ob, gt = env1.ob, env1.gt
                ob = ob[:, np.newaxis, :]  # Add batch axis
                inputs = torch.from_numpy(ob).type(torch.float).to(device)
    
                action_pred = net(inputs)
                action_pred = action_pred.cpu().detach().numpy()
                action_pred = np.argmax(action_pred, axis=-1)
                perf += gt[-1] == action_pred[-1, 0]
    
            perf /= num_trial
            performances_task1.append(perf)
            
            print('{:d} loss: {:0.5f}   Performance(T1): {:0.5f}'.format(i + 2001, running_loss / 200, perf))
            running_loss = 0.0   
    
            #Performance test on Task 2
            perf = 0
            for j in range(num_trial):
                
                env2.new_trial()
                ob, gt = env2.ob, env2.gt
                ob = ob[:, np.newaxis, :]  # Add batch axis
                inputs = torch.from_numpy(ob).type(torch.float).to(device)
    
                action_pred = net(inputs)
                action_pred = action_pred.cpu().detach().numpy()
                action_pred = np.argmax(action_pred, axis=-1)
                perf += gt[-1] == action_pred[-1, 0]
    
            perf /= num_trial
            performances_task2.append(perf)
            
            print('                    Performance(T2): {:0.5f}'.format(perf))
            print('')
            running_loss = 0.0
    all_performances_task1.append(performances_task1)
    all_performances_task2.append(performances_task2)

print(np.asarray(all_performances_task1), np.asarray(all_performances_task2))
np.save('./data/training-task2task1_no-replay_performance-task1', all_performances_task1)
np.save('./data/training-task2task1_no-replay_performance-task2', all_performances_task2)

    #Add to the matrix.