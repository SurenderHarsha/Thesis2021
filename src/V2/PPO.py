#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import math
import random
import argparse

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init,hidden_size = 512,num_channels = 5):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        self.actor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(4,2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,1),
            #nn.Conv2d(128, 256, kernel_size=4, stride=1),
            #nn.BatchNorm2d(256),
            #nn.MaxPool2d(2,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(124416, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,action_dim),
            nn.Tanh()
            
            
        )
        
        self.critic = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(4,2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,1),
            #nn.Conv2d(128, 256, kernel_size=4, stride=1),
            #nn.BatchNorm2d(256),
            #nn.MaxPool2d(2,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(124416, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1),
            nn.Tanh()
            
            
        )
        
        
        #self.l1 = nn.Conv2d(num_channels, 32, kernel_size=4, stride=2)
        #self.l2 = nn.Conv2d(32, 16, kernel_size=4, stride=2)
        #self.l3 = nn.Conv2d(16, 8, kernel_size=3, stride=1)
        #self.p = nn.MaxPool2d(2, 2)
        ''''
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=4, stride=2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, kernel_size=4, stride=2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            
            #nn.MaxPool2d(2, 2)
        )
        
        self.l1 = nn.ConvTranspose2d(16, 16, kernel_size = 8, stride=4)
        self.relu = nn.ReLU()
        self.l2 = nn.ConvTranspose2d(16, 32, kernel_size = 8, stride=1)
        self.l3 = nn.ConvTranspose2d(32, 3, kernel_size = 8, stride=2)
        self.sig = nn.Sigmoid()
        
        self.layer1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=4)
        self.relu = nn.ReLU();
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.layer3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.layer4 = nn.Linear(9120,512)
        self.actol = nn.Linear(512, action_dim)
        self.acto = nn.Tanh()
        self.crit = nn.Linear(512, 1)
        '''
        #self.actor = nn.Sequential(self.main,nn.Linear(hidden_size, action_dim))
        #self.critic = nn.Sequential(self.main_two,nn.Linear(hidden_size, 1))
        '''
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        '''
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def AutoEncoder(self,x):
        '''
        x = self.encoder(x)
        print(x.shape)
        x = self.l1(x)
        x = self.relu(x)
        print(x.shape)
        x = self.l2(x)
        x = self.relu(x)
        print(x.shape)
        x = self.l3(x)
        x = self.sig(x)
        print(x.shape)
        
        print(x.shape)
        x = self.l1(x)
        print(x.shape)
        x = self.p(x)
        print(x.shape)
        x = self.l2(x)
        print(x.shape)
        x = self.p(x)
        print(x.shape)
        x = self.l3(x)
        print(x.shape)
        x = self.p(x)
        print(x.shape)
        '''
        #x = self.encoder(x)
        #return x
        pass
        

    def forward(self):
        raise NotImplementedError
        '''
        print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        a = self.actol(x)
        a = self.acto(a)
        v = self.crit(x)
        return a,v
        '''
        
    def backward(self, x):
        import pdb
        pdb.set_trace()
        return x
    

    def act(self, state):

        if self.has_continuous_action_space:
            #x = self.main(state)
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            #dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            if len(state.shape) == 3:
                state = state.reshape((1,5,186,150))
            #x = self.main(state)
            #print(state.shape)
            action_mean = self.actor(state)
            #print(action_mean.shape)
            action_var = self.action_var.expand_as(action_mean)
            #print(action_var,action_var.shape)
            cov_mat = torch.diag_embed(action_var).to(device)
            #print(cov_mat)
            dist = MultivariateNormal(action_mean, cov_mat)
            #print(dist)
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            #dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        #print(state_values,action_mean)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        
                        {'params': self.policy.critic.parameters(),'lr': lr_critic}
                        
                    ])
        self.optimizer2 = torch.optim.Adam([
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        #print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                #print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                pass
                #print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        #print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        #print(self.buffer.rewards)
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        #print("I",rewards)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        #print(rewards,rewards.mean(),rewards.std(unbiased = False))
        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased = False) + 1e-7)
        #print(rewards)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            #print(state_values)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            #state_values = torch.
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            #print(ratios)
            # Finding Spurrogate Loss
            advantages = rewards - state_values.detach()   
            #print(rewards)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            #print(state_values)
            #print(state_values.shape)
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            #print(surr1,surr2,state_values,rewards,dist_entropy,loss)
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            #self.optimizer2.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))