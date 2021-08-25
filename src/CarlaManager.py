#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Set your NGSIM, openDD and Carla Path here.
#os.environ["NGSIM_DIR"] = "/home/surender/Downloads/NGSIM"
#os.environ["OPENDD_DIR"] = "/home/surender/Downloads/openDD"
#os.environ["CARLA_PATH"] = "/home/surender/Downloads/carlaOld"

nps = "/home/surender/Downloads/NGSIM"
ops = "/home/surender/Downloads/openDD"
cps = "/home/surender/Downloads/carlaOld"

import sys
#sys.path.append('/home/surender/Downloads/CARLA_0.9.9.4/PythonAPI/carla/dist')
import carla
import random
import argparse

from carla_real_traffic_scenarios.carla_maps import CarlaMaps
from carla_real_traffic_scenarios.ngsim import NGSimDatasets, DatasetMode
from carla_real_traffic_scenarios.ngsim.scenario import NGSimLaneChangeScenario
from carla_real_traffic_scenarios.opendd.scenario import OpenDDScenario
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import Scenario

from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

from PIL import Image
from IPython.display import clear_output, Image, display, HTML
import cv2

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import math
import pickle

from VehicleActor import *
from LearningAlgorithm import *
#import A3C_Algorithm
from A3C_Algorithm import *

import warnings
warnings.filterwarnings("ignore")

def prepare_ngsim_scenario(client: carla.Client) -> Scenario:
    data_dir = os.environ.get("NGSIM_DIR")
    #data_dir = os.listdir('/home/surender/Downloads/NGSIM')
    assert data_dir, "Path to the directory with NGSIM dataset is required"
    ngsim_map = NGSimDatasets.list()
    ngsim_dataset = ngsim_map[1]
    client.load_world(ngsim_dataset.carla_map.level_path)
    return NGSimLaneChangeScenario(
        ngsim_dataset,
        dataset_mode=DatasetMode.TRAIN,
        data_dir=data_dir,
        reward_type=RewardType.DENSE,  #This is where the reward type is defined.
        client=client,
    )

#Setting up the actor of the main vehicle.
def prepare_ego_vehicle(world: carla.World) -> carla.Actor:
    car_blueprint = world.get_blueprint_library().find("vehicle.audi.a2")

    # This will allow external scripts like manual_control.py or no_rendering_mode.py
    # from the official CARLA examples to take control over the ego agent
    car_blueprint.set_attribute("role_name", "hero")

    # spawn points doesnt matter - scenario sets up position in reset
    ego_vehicle = world.spawn_actor(
        car_blueprint, carla.Transform(carla.Location(0, 0, 500), carla.Rotation())
    )

    assert ego_vehicle is not None, "Ego vehicle could not be spawned"

    # Setup any car sensors you like, collect observations and then use them as input to your model
    return ego_vehicle

class CarlaManager(object):
    
    def __init__(self,NGSIM_path,openDD_path,Carla_path,host ="localhost",port = 2000):
        os.environ["NGSIM_DIR"] = NGSIM_path
        os.environ["OPENDD_DIR"] = openDD_path
        os.environ["CARLA_PATH"] = Carla_path
        self.carla_path = Carla_path
        
        self.host = host
        self.port = port
        self.hp = None
        self.history = {'step': [],
                        'score': [],
                        'theta': []}
        #self.generate_theta = False
        self.historical_steps = 0
        self.policy = None
        try:
            self.carla_simulator = threading.Thread(target = self.cmd_carla)
            self.carla_simulator.start()
            self.delay(5)
            
            self.client = carla.Client(host,port)
            self.scenario = prepare_ngsim_scenario(self.client)
            self.world = self.client.get_world()
            self.spectator = self.world.get_spectator()
            self.ego_vehicle = prepare_ego_vehicle(self.world)
            self.birdview_producer = BirdViewProducer(
            self.client,  # carla.Client
            target_size=PixelDimensions(width=150, height=336),
            pixels_per_meter=4,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
            )
        except Exception as e:
            print(e)
            self.reload_carla()
            
            
        
    def reload_carla(self):
        try:
            self.carla_simulator = threading.Thread(target = self.cmd_carla)
            self.carla_simulator.start()
           
            print("Restarting Carla after crash....")
            self.delay(5)
            
            self.client = carla.Client(self.host,self.port)
            self.scenario = prepare_ngsim_scenario(self.client)
            self.world = self.client.get_world()
            self.spectator = self.world.get_spectator()
            self.ego_vehicle = prepare_ego_vehicle(self.world)
            self.birdview_producer = BirdViewProducer(
            self.client,  # carla.Client
            target_size=PixelDimensions(width=150, height=336),
            pixels_per_meter=4,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
            )
        except:
            print("Crash failed! reloading until fixed")
            #self.reload_carla()
        
        
    def delay(self,seconds):
        time.sleep(seconds)
        
    def cmd_carla(self):
        os.system("DISPLAY= "+self.carla_path+"/CarlaUE4.sh -benchmark -fps=10 -quality-level=Low -opengl -Resx=300 -Resy=300 -NoVSync -carla-streaming-port=0 &>> log.txt")
    
    
    def ARS_Algorithm(self):
        pass
    
    def AC3_Algorithm(self):
        pass
    
    def simple_radar_ars(self,input_size,output_size):
        def explore(direction=None, delta=None):
            try:
                self.scenario.reset(self.ego_vehicle)
                self.world.tick()
                manager = VehicleManager(self.ego_vehicle,self.world,self.scenario,baseline=True)
                done = False
                #state = self.env.reset()
                #done = False
                sum_rewards = 0.0
                steps = 0
                self.steering_cache = []
                while not done:
                    # Get data from front camera and divide by 255 to normalize
                    #state = self.env.front_camera.reshape(1, 224, 224, 3) / 255.
                    manager.radar_vectors(30)
                    a_t,a_s = manager.ars_data()
                    state = np.array([a_t,a_s])
                    steps += 1
                    
                    action = self.policy.evaluate(state, delta, direction)
                    '''
                    if action[0] >0.7:
                        sum_rewards -= action[0]
                    if action[1] <= -0.9 or action[1] >= 0.9:
                        sum_rewards -= 1
                    '''
                    self.ego_vehicle.apply_control(carla.VehicleControl(throttle=np.clip(action[0], 0.0, 1.0), steer=np.clip(action[1], -1.0, 1.0),brake=np.clip(action[2], 0.0, 1.0)))
                    cmd, reward, done, _ = self.scenario.step(self.ego_vehicle)
                    #reward *= 20
                    self.steering_cache.append(action[1])
                    v = self.ego_vehicle.get_velocity()
                    kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
                    
                    if kmh < 60 & kmh > 0.2:
                        #done = False
                        reward += 1 #-1
                        # Reward lighter steering when moving
                        if np.abs(action[1]) < 0.3:
                            reward += 9
                        elif np.abs(action[1]) > 0.5 and np.abs(action[1]) < 0.9:
                            reward -= 1
                        elif np.abs(action[1]) >= 0.9:
                            reward -= 6
                    elif kmh < 0.2:
                        reward -= 10
                    else:
                        reward += 20
                        if np.abs(action[1]) < 0.3:
                            reward += 20
                        # Reduce score for heavy steering
                        if np.abs(action[1]) > 0.5 and np.abs(action[1]) < 0.9:
                            reward -= 10
                        elif np.abs(action[1]) >= 0.9:
                            reward -= 20
                    reward -= (np.abs(np.mean(self.steering_cache)) + np.abs(action[1])) * 10 / 2
                    #state, reward, done, _ = self.env.step(action, steps)
                    #reward = max(min(reward, 1), -1)
                    sum_rewards += reward
                    self.world.tick()
                    #self.delay(0.1)
                #sum_rewards += steps
                print('Worker saw {} steps'.format(steps))
                # Average the rewards per step to account for variable FPS seen by workers
                print('Sum of episode rewards:', sum_rewards)
                if steps == 0:
                    steps = 1
                adjusted_reward = sum_rewards / steps
                print('Adjusted Reward for episode:', adjusted_reward)
                del manager
                
                return adjusted_reward
            except Exception as e:
                print(e,"Reload Carla")
                self.reload_carla()

        def train_radar(render = False):
            

            
            for step in range(self.hp.nb_steps):
                self.historical_steps += 1
                print('Performing step {}. ({}/{})'.format(self.historical_steps,
                                                           step + 1,
                                                           self.hp.nb_steps
                                                          ))
                start = time.time()
                # Only record video during evaluation, every n steps
                if render or step % self.hp.record_every == 0:
                        
                        birdview = self.birdview_producer.produce(
                        agent_vehicle=self.ego_vehicle  # carla.Actor (spawned vehicle)
                        )
                        rgb = BirdViewProducer.as_rgb(birdview)
                        cv2.imshow('Frame',rgb)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                # initialize the random noise deltas and the positive/negative rewards
                deltas = self.policy.sample_deltas()
                positive_rewards = [0] * self.hp.num_deltas
                negative_rewards = [0] * self.hp.num_deltas
    
                # play an episode each with positive deltas and negative deltas, collect rewards
                for k in range(self.hp.num_deltas):
                    positive_rewards[k] = explore(direction="+", delta=deltas[k])
                    negative_rewards[k] = explore(direction="-", delta=deltas[k])
                try:
                # Compute the standard deviation of all rewards
                    sigma_rewards = np.array(positive_rewards + negative_rewards).std()
        
                    # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
                    scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
                    order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:self.hp.num_best_deltas]
                    rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
                    # Update the policy
                    self.policy.update(rollouts, sigma_rewards)
                except:
                    print("This update failed. Retry!")
    
                if step % self.hp.record_every == 0:
                    # Play an episode with the new weights and print the score
                    reward_evaluation = explore()
                    print('Step:', step + 1, 'Reward:', reward_evaluation)
                    self.history['step'].append(self.historical_steps)
                    self.history['score'].append(reward_evaluation)
                    self.history['theta'].append(self.policy.theta.copy())
                    with open('history_save'+str(step)+'.pickle', 'wb') as handle:
                        pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    #self.save()
                    
                end = time.time()
                print('Time to complete update step:', end - start)
                #self.env.show_cam = False
            cv2.destroyAllWindows()
        self.hp = HP(nb_steps=40, 
             noise=0.1, 
             learning_rate=0.05, 
             num_deltas=8, 
             num_best_deltas=4,
             record_every=5
            )
        self.policy = Policy(input_size, output_size, self.hp)
        train_radar()
        
    
    def simple_radar_ars_imitation(self,input_size,output_size):
        def explore(direction=None, delta=None, train_step = None):
            try:
                chauffer_command = 0
                self.scenario.reset(self.ego_vehicle)
                self.world.tick()
                manager = VehicleManager(self.ego_vehicle,self.world,self.scenario,baseline=True)
                done = False
                #state = self.env.reset()
                #done = False
                sum_rewards = 0.0
                steps = 0
                self.steering_cache = []
                while not done:
                    # Get data from front camera and divide by 255 to normalize
                    #state = self.env.front_camera.reshape(1, 224, 224, 3) / 255.
                    manager.radar_vectors(30)
                    l = manager.ars_data_two()
                    l.append(chauffer_command)
                    
                    state = np.array(l)
                    #state = np.array([chauffer_command,self.ego_vehicle.get_location().x,self.ego_vehicle.get_location().y])
                    steps += 1
                    
                    action = self.policy.evaluate(state, delta, direction)
                    '''
                    if action[0] >0.7:
                        sum_rewards -= action[0]
                    if action[1] <= -0.9 or action[1] >= 0.9:
                        sum_rewards -= 1
                    '''
                    self.ego_vehicle.apply_control(carla.VehicleControl(throttle=np.clip(action[0], 0.0, 1.0), steer=np.clip(action[1], -1.0, 1.0),brake=np.clip(action[2], 0.0, 1.0)))
                    cmd, reward, done, _ = self.scenario.step(self.ego_vehicle)
                    chauffer_command = cmd.value
                    #reward *= 20
                    self.steering_cache.append(action[1])
                    v = self.ego_vehicle.get_velocity()
                    kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
                    reward *= 60
                    if reward>=60:
                        print("Scenario COMPLETED SUCCESSFULLY!")
                        return 1000
                    if kmh < 60 & kmh > 0.2:
                        #done = False
                        reward += 1 #-1
                        # Reward lighter steering when moving
                        if np.abs(action[1]) < 0.3:
                            reward += 9
                        elif np.abs(action[1]) > 0.5 and np.abs(action[1]) < 0.9:
                            reward -= 1
                        elif np.abs(action[1]) >= 0.9:
                            reward -= 6
                    elif kmh < 0.2:
                        reward -= 10
                    else:
                        reward += 20
                        if np.abs(action[1]) < 0.3:
                            reward += 20
                        # Reduce score for heavy steering
                        if np.abs(action[1]) > 0.5 and np.abs(action[1]) < 0.9:
                            reward -= 10
                        elif np.abs(action[1]) >= 0.9:
                            reward -= 20
                    reward -= (np.abs(np.mean(self.steering_cache)) + np.abs(action[1])) * 10 / 2
                    #state, reward, done, _ = self.env.step(action, steps)
                    #reward = max(min(reward, 1), -1)
                    sum_rewards += reward
                    self.world.tick()
                    #self.delay(0.1)
                #sum_rewards += steps
                print('Worker saw {} steps'.format(steps))
                # Average the rewards per step to account for variable FPS seen by workers
                print('Sum of episode rewards:', sum_rewards)
                if steps == 0:
                    steps = 1
                adjusted_reward = sum_rewards / steps
                print('Adjusted Reward for episode:', adjusted_reward)
                del manager
                
                return adjusted_reward
                
            except Exception as e:
                print(e,"Reload Carla")
                self.reload_carla()

        def train_radar(render = False):
            

            
            for step in range(self.hp.nb_steps):
                self.historical_steps += 1
                print('Performing step {}. ({}/{})'.format(self.historical_steps,
                                                           step + 1,
                                                           self.hp.nb_steps
                                                          ))
                start = time.time()
                # Only record video during evaluation, every n steps
                if render or step % self.hp.record_every == 0:
                        
                        birdview = self.birdview_producer.produce(
                        agent_vehicle=self.ego_vehicle  # carla.Actor (spawned vehicle)
                        )
                        rgb = BirdViewProducer.as_rgb(birdview)
                        cv2.imshow('Frame',rgb)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                # initialize the random noise deltas and the positive/negative rewards
                deltas = self.policy.sample_deltas()
                positive_rewards = [0] * self.hp.num_deltas
                negative_rewards = [0] * self.hp.num_deltas
    
                # play an episode each with positive deltas and negative deltas, collect rewards
                for k in range(self.hp.num_deltas):
                    positive_rewards[k] = explore(direction="+", delta=deltas[k],train_step = step)
                    negative_rewards[k] = explore(direction="-", delta=deltas[k],train_step = step)
                try:
                # Compute the standard deviation of all rewards
                    sigma_rewards = np.array(positive_rewards + negative_rewards).std()
        
                    # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
                    scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
                    order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:self.hp.num_best_deltas]
                    rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
                    # Update the policy
                    self.policy.update(rollouts, sigma_rewards)
                except:
                    print("This update failed. Retry!")
    
                if step % self.hp.record_every == 0:
                    # Play an episode with the new weights and print the score
                    reward_evaluation = explore()
                    print('Step:', step + 1, 'Reward:', reward_evaluation)
                    self.history['step'].append(self.historical_steps)
                    self.history['score'].append(reward_evaluation)
                    self.history['theta'].append(self.policy.theta.copy())
                    with open('history_cmd'+str(step)+'.pickle', 'wb') as handle:
                        pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    #self.save()
                    
                end = time.time()
                print('Time to complete update step:', end - start)
                #self.env.show_cam = False
            cv2.destroyAllWindows()
        self.hp = HP(nb_steps=40, 
             noise=0.1, 
             learning_rate=0.05, 
             num_deltas=8, 
             num_best_deltas=4,
             record_every=5
            )
        self.policy = Policy(input_size, output_size, self.hp)
        train_radar()
    
        
    #Template for training
    def gym_iteration(self,epochs = 50,render = False,policy = None):
        for epoch in range(epochs):
            try:
                self.scenario.reset(self.ego_vehicle)
                self.world.tick()
                manager = VehicleManager(self.ego_vehicle,self.world,self.scenario)
                done = False
                iterations = 0
                manager.update_neighbours()
                manager.update_buffers()
                self.delay(3)
                sum_rewards = 0
                while not done:
                    
                    #Here we get policy output
                    manager.update_buffers()
                    e,n = manager.return_data()
                    throttle = np.random.rand()
                    steer = np.random.rand()*2 - 1
                    
                    
                    if render:
                        
                        birdview = self.birdview_producer.produce(
                        agent_vehicle=self.ego_vehicle  # carla.Actor (spawned vehicle)
                        )
                        rgb = BirdViewProducer.as_rgb(birdview)
                        cv2.imshow('Frame',rgb)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    
                    self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
                    cmd, reward, done, _ = self.scenario.step(self.ego_vehicle)
                    sum_rewards+=reward
                    iterations +=1
                    self.world.tick()
                    self.delay(0.1)
                del manager
                cv2.destroyAllWindows()
                print(sum_rewards)
            except Exception as e:
                print(e,"Reload Carla")
                self.reload_carla()
                
                
                
    def run_basic_a3c(self,epochs = 50,render = False):
        pass
    
    
class Worker(mp.Process):
     
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                if self.name == 'w0':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a.clip(-2, 2))
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)       