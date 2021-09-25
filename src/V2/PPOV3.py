#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal,Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
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
        
        self.layer1 = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Dropout(0.4),
            nn.Linear(512,128)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(131, 64),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.Dropout(0.4),
            nn.ReLU()
        
        )
        self.actor = nn.Sequential(
            
            nn.Linear(64,1),
            nn.Softplus()
        )
        self.critic = nn.Sequential(
            nn.Linear(131, 64),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            
            nn.Linear(64,1),
            nn.Softplus()
        )
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
        
    def backward(self, x):
        import pdb
        pdb.set_trace()
        return x
    

    def act(self, state):

        if self.has_continuous_action_space:
            #x = self.main(state)
            inps1 = state[:,:-3]
            inps2 = state[:,-3:]
            action_m = self.layer1(inps1)
            #print(action_m,inps2,action_m.shape,inps2.shape)
            inps = torch.cat((action_m,inps2),1)
            x = self.layer2(inps)
            action_mean = self.actor(x) + 0.001
            action_var = self.var(x) + 0.001
            dist = Beta(action_mean, action_var)
            
        else:
            action_probs = self.actor(state)
            #dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            if len(state.shape) == 3:
                state = state.reshape((1,5,224,224))
            inps1 = state[:,:-3]
            inps2 = state[:,-3:]
            action_m = self.layer1(inps1)
            inps = torch.cat((action_m,inps2),1)
            x = self.layer2(inps)
            action_mean = self.actor(x) + 0.001
            action_var = self.var(x) + 0.001
            dist = Beta(action_mean, action_var)
            #print(dist)
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            #dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        #dist_entropy = dist.entropy()
        inps1 = state[:,:-3]
        inps2 = state[:,-3:]
        critic_m = self.layer1(inps1)
        inps = torch.cat((critic_m,inps2),1)
        state_values = self.critic(inps)
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
                        
                        {'params': self.policy.critic.parameters(),'lr': lr_critic},
                        {'params': self.policy.layer1.parameters(),'lr': lr_actor},
                        {'params': self.policy.layer2.parameters(),'lr': lr_actor},
                        {'params': self.policy.var.parameters(),'lr': lr_actor}
                        
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


    def select_action(self, state,validation = False):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            if not validation:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
            else:
                del state
                del action_logprob
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
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer.states))), 32, False):
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[index], old_actions[index])
                #print(state_values)
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                #state_values = torch.
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs[index].detach())
                #print(ratios)
                # Finding Spurrogate Loss
                advantages = rewards[index] - state_values.detach()   
                #print(rewards)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                #print(state_values)
                #print(state_values.shape)
                # final loss of clipped objective PPO
                
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards[index]) - 0.01*dist_entropy
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
