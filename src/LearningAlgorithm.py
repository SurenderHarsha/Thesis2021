#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code from ARS Carla

import numpy as np


#https://github.com/FoamoftheSea/mod6project 
class HP():
    # Hyperparameters
    def __init__(self,
                 nb_steps=100,
                 episode_length=2000,
                 learning_rate=0.02,
                 num_deltas=4,
                 num_best_deltas=2,
                 noise=0.03,
                 seed=1,
                 
                 record_every=50):

        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise
        self.seed = seed
        #self.env_name = env_name
        self.record_every = record_every
        
        

# Skow's Policy class
class Policy():
    def __init__(self, input_size, output_size, hp, theta=None):
        self.input_size = input_size
        self.output_size = output_size
        if theta is not None:
            self.theta = theta
        else:
            #self.theta = np.random.random((output_size, input_size))
            self.theta = np.zeros((output_size, input_size))
        self.hp = hp

    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + self.hp.noise * delta).dot(input)
        elif direction == "-":
            return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]

    def update(self, rollouts, sigma_rewards):
        # sigma_rewards is the standard deviation of the rewards
        old_theta = self.theta.copy()
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        theta_update = self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step
        self.theta += theta_update
        if np.array_equal(old_theta, self.theta):
            print("Theta did not change.")
            
