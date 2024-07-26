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
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import MaxNLocator

# Defines the environment.
task = 'PerceptualDecisionMaking-v0'
task2 = 'PerceptualDecisionMakingDelayResponse-v0'

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

simulations = []
all_performances_task1_training1 = np.empty(0)
all_performances_task2_training1 = np.empty(0)
all_performances_task1_training2 = np.empty(0)
all_performances_task2_training2 = np.empty(0)

for s in range(1):
    
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
    
    s += 1
    running_loss = 0.0
    x = []
    ticks = []
    losses = []
    performances_task1_training1 = []
    performances_task2_training1 = []

    print("Simulation:", s)
    print("Task 2 training:")
    
    #Main training loop
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
            num_trial = 200
            
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
            performances_task2_training1.append(perf)
            
            print('{:d} loss: {:0.5f}   Performance(T2): {:0.5f}'.format(i + 1, running_loss / 200, perf))
            running_loss = 0.0   
            
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
            performances_task1_training1.append(perf)
            
            print('                    Performance(T1): {:0.5f}'.format(perf))
            print('')
            running_loss = 0.0                
         
        if i % 200 == 199:
            ticks.append(i+1)
            
    #Graph the loss and performances during training 1:
    
    #Loss during training 1
    loss_training1,axes = plt.subplots()
    axes.plot(x,losses,marker = 'o')
    axes.set_xticks(ticks)
    axes.xaxis.set_major_formatter(EngFormatter())
    axes.set_title("Loss Throughout Task 2 Training in (Simulation {:d})".format(s), fontsize = 18)
    axes.set_xlabel("Iterations", fontsize = 15)
    axes.set_ylabel("Loss", fontsize = 15)
    
    #Graph the performances during training 2.
    performancesgraph_training1,axes = plt.subplots()
    axes.plot(x,performances_task1_training1,marker = 'o', label = 'Task 1', color = 'green')
    axes.plot(x,performances_task2_training1,marker = 'o', label = 'Task 2', color = 'magenta')
    axes.set_xticks(ticks)
    axes.xaxis.set_major_formatter(EngFormatter())
    axes.set_title("Performances Throughout Task 2 Training in (Simulation {:d})".format(s), fontsize = 18)
    axes.set_xlabel("Iterations", fontsize = 15)
    axes.set_ylabel("Performance", fontsize = 15)
    axes.legend()
    print('')
    
#Testing 1----------------------------------------------------------------------   

    #Test of task 1 after training 1-------------------------------------------------------------------------
    
    print("Simulation:", s)
    perf = 0
    for i in range(num_trial):
        env1.new_trial()
        ob, gt = env1.ob, env1.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)
    
        action_pred = net(inputs)
        action_pred = action_pred.cpu().detach().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        perf += gt[-1] == action_pred[-1, 0]
    
    perf /= num_trial
    print('Performance on task 1 ("PerceptualDecisionMaking-v0") in {:d} trials:'.format(num_trial))
    print(perf)
    all_performances_task1_training1 = np.append(all_performances_task1_training1, perf)

    #Test of task 2 after training 1-------------------------------------------

    perf = 0
    for i in range(num_trial):
        env2.new_trial()
        ob, gt = env2.ob, env2.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)
    
        action_pred = net(inputs)
        action_pred = action_pred.cpu().detach().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        perf += gt[-1] == action_pred[-1, 0]
    
    perf /= num_trial
    print('Performance on task 2 ("PerceptualDecisionMakingDelayResponse-v0") in {:d} trials:'.format(num_trial))
    print(perf)
    print('')
    all_performances_task2_training1 = np.append(all_performances_task2_training1, perf)
 
#Training 2----------------------------------------------------------------------
    
    x2 = []
    ticks = []
    losses = []
    performances_task1_training2 = []
    performances_task2_training2 = []
    print("Task 1 training:")
    
    #Main training loop
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
            
            x2.append(i + 2001)
            losses.append(running_loss / 200)
    
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
            performances_task2_training2.append(perf)
            
            print('{:d} loss: {:0.5f}   Performance(T2): {:0.5f}'.format(i + 2001, running_loss / 200, perf))
            running_loss = 0.0   
    
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
            performances_task1_training2.append(perf)
            
            print('                    Performance(T1): {:0.5f}'.format(perf))
            print('')
            running_loss = 0.0    
            
        if i % 200 == 199:
            ticks.append(2000 + 1 + i)
    
    #Graph the loss and performances during training 2:      
    
    #Loss during training 2
    loss_task1,axes = plt.subplots()
    axes.plot(x2,losses,marker = 'o')
    axes.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
    axes.set_xticks(ticks)
    axes.xaxis.set_major_formatter(EngFormatter())
    axes.set_title("Loss Throughout Task 1 Training in (Simulation {:d})".format(s), fontsize = 18)
    axes.set_xlabel("Iterations", fontsize = 15)
    axes.set_ylabel("Loss", fontsize = 15)
    
    #Graph the performances during training 2.
    performancesgraph_training2,axes = plt.subplots()
    axes.plot(x2,performances_task1_training2,marker = 'o', label = 'Task 1', color = 'green')
    axes.plot(x2,performances_task2_training2,marker = 'o', label = 'Task 2', color = 'magenta')
    axes.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
    axes.set_xticks(ticks)
    axes.xaxis.set_major_formatter(EngFormatter())
    axes.set_title("Performances Throughout Task 1 Training in (Simulation {:d})".format(s), fontsize = 18)
    axes.set_xlabel("Iterations", fontsize = 15)
    axes.set_ylabel("Performance", fontsize = 15)
    axes.legend()
    print('')

#Testing 2---------------------------------------------------------------------    
    
    #Test of task 1 after training 2-------------------------------------------------------------------------
    
    #Change the environment to task 1 ("PerceptualDecisionMaking-v0")
    
    print("Simulation:", s)
    perf = 0
    for i in range(num_trial):
        env1.new_trial()
        ob, gt = env1.ob, env1.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)
    
        action_pred = net(inputs)
        action_pred = action_pred.cpu().detach().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        perf += gt[-1] == action_pred[-1, 0]
    
    perf /= num_trial
    print('Final performance on task 1 ("PerceptualDecisionMaking-v0") in {:d} trials:'.format(num_trial))
    print(perf)
    all_performances_task1_training2 = np.append(all_performances_task1_training2, perf)
    
#Test of task 2 after training 2 -------------------------------------------------------------------------
    
    #Change the environment to task 2 ("PerceptualDecisionMakingDelayResponse-v0")
    
    perf = 0
    for i in range(num_trial):
        env2.new_trial()
        ob, gt = env2.ob, env2.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)
    
        action_pred = net(inputs)
        action_pred = action_pred.cpu().detach().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        perf += gt[-1] == action_pred[-1, 0]
    
    perf /= num_trial
    print('Final performance on task 2 ("PerceptualDecisionMakingDelayResponse-v0") in {:d} trials:'.format(num_trial))
    print(perf)
    print('')
    all_performances_task2_training2 = np.append(all_performances_task2_training2, perf)
    simulations.append(s)
    x2.insert(0,2000)
    performances_task1_training2.insert(0,performances_task1_training1[-1])
    performances_task2_training2.insert(0,performances_task2_training1[-1])
    
    performances_of_simulation, axes = plt.subplots()
    axes.plot(x,performances_task1_training1, color = 'green', marker = 'o', label = 'Task 1')
    axes.plot(x,performances_task2_training1, color = 'magenta', marker = 'o', label = 'Task 2')
    axes.axhline(y = 0.5, color = 'black', linestyle = 'dashed')
    axes.axvline(x = 2000, color = 'black', linestyle = 'dotted')
    axes.plot(x2,performances_task1_training2, color = 'green', marker = 'o', label = 'Task 1')
    axes.plot(x2,performances_task2_training2, color = 'magenta', marker = 'o', label = 'Task 2')
    axes.axvline(x = 2000, color = 'black', linestyle = 'dotted')
    axes.set_xticks([1000,2000,3000,4000])
    axes.set_yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
    axes.set_title("Task 2 Training Before Task 1 Training (no replays)", fontsize = 18)
    axes.xaxis.set_major_formatter(EngFormatter())
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    axes.set_xlabel("Iterations", fontsize = 15)
    axes.set_ylabel("Performance", fontsize = 15)
    axes.legend(['Task 1', 'Task 2', 'Chance', 'Transition'], loc = 'lower right', prop = {'size':13})
    performances_of_simulation.savefig('C:/Users/austi/Downloads/Project/Graphs/Task 2 Training Before Task 1 Training (no replays).svg')
#------------------------------------------------------------------------------ 

#Final Analysis

if s > 1:

#Calculates and prints the average of the performance across all simulations.
    averageperformance_task1_training1 = np.mean(all_performances_task1_training1,0)
    averageperformance_task2_training1 = np.mean(all_performances_task2_training1,0)
    averageperformance_task1_training2 = np.mean(all_performances_task1_training2,0)
    averageperformance_task2_training2 = np.mean(all_performances_task2_training2,0)
    
    print("Average performance across all trials of task 1 after training 1:", averageperformance_task1_training1)
    print("Average performance across all trials of task 2 after training 1:", averageperformance_task2_training1)
    print("Average performance across all trials of task 1 after training 2:", averageperformance_task1_training2)
    print("Average performance across all trials of task 2 after training 2:", averageperformance_task2_training2)
    
    #Graphs the performances of each simulations.
    averageperformances_graph,axes = plt.subplots()
    axes.plot(simulations,all_performances_task1_training1, color = 'green', marker = 'o', label = 'Task 1 Training 1')
    axes.plot(simulations,all_performances_task2_training1, color = 'magenta', marker = 'o', label = 'Task 2 Training 1')
    axes.plot(simulations,all_performances_task1_training2, color = 'green', marker = 'o', label = 'Task 1 Training 2')
    axes.plot(simulations,all_performances_task2_training2, color = 'magenta', marker = 'o', label = 'Task 2 Training 2')
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_title("Performances Across All Simulations", fontsize = 18)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    axes.set_xlabel("Simulation #", fontsize = 15)
    axes.set_ylabel("Performance", fontsize = 15)
    axes.legend()
    print('')
