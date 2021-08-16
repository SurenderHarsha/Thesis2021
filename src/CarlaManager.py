#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Set your NGSIM, openDD and Carla Path here.
#os.environ["NGSIM_DIR"] = "/home/surender/Downloads/NGSIM"
#os.environ["OPENDD_DIR"] = "/home/surender/Downloads/openDD"
#os.environ["CARLA_PATH"] = "/home/surender/Downloads/carlaOld"

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

class CarlaManager(object):
    
    def __init__(self,NGSIM_path,openDD_path,Carla_path,host ="localhost",port = 2000):
        os.environ["NGSIM_DIR"] = NGSIM_path
        os.environ["OPENDD_DIR"] = openDD_path
        os.environ["CARLA_PATH"] = Carla_path
        
        self.host = host
        self.port = port
        
        self.client = carla.Client(host,port)
        self.scenario = prepare_ngsim_scenario(client)
        self.world = client.get_world()
        self.spectator = world.get_spectator()
        self.ego_vehicle = prepare_ego_vehicle(world)
        self.birdview_producer = BirdViewProducer(
        client,  # carla.Client
        target_size=PixelDimensions(width=150, height=336),
        pixels_per_meter=4,
        crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
        )
        
        