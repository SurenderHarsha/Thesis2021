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

from VehicleActor import *

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
        