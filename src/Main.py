# -*- coding: utf-8 -*-
import os

# Set your NGSIM, openDD and Carla Path here.
os.environ["NGSIM_DIR"] = "/home/surender/Downloads/NGSIM"
os.environ["OPENDD_DIR"] = "/home/surender/Downloads/openDD"
os.environ["CARLA_PATH"] = "/home/surender/Downloads/carlaOld"

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

#Start Carla, Set carla path here after installing.
def cmd_carla():
    os.system("DISPLAY= /home/surender/Downloads/carlaOld/CarlaUE4.sh -benchmark -fps=10 -quality-level=Low -opengl -Resx=300 -Resy=300 -NoVSync")

#Load the lane change scenario
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

def reload_carla():
    global host,port,client,scenario, world,spectator,ego_vehicle,birdview_producers,p
    p = threading.Thread(target = cmd_carla)
    p.start()
    print("Restarting carla.....")
    time.sleep(5)
    
    
    print("Connecting to carla...")
    host = "localhost"
    port = 2000
    client = carla.Client(host,port)
    scenario = prepare_ngsim_scenario(client)
    world = client.get_world()
    spectator = world.get_spectator()
    ego_vehicle = prepare_ego_vehicle(world)
    birdview_producer = BirdViewProducer(
    client,  # carla.Client
    target_size=PixelDimensions(width=150, height=336),
    pixels_per_meter=4,
    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
    )
    ldr_bp = world.get_blueprint_library().find('sensor.other.radar')
    ldr_loc = carla.Location(0,0,0)
    ldr_rot = carla.Rotation(0,0,0)
    ldr_bp.set_attribute("range",'10.0')


    ldr_transform = carla.Transform(ldr_loc,ldr_rot)
    ego_ldr = world.spawn_actor(ldr_bp,ldr_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
    ego_ldr.listen(lambda dat: load_radar(dat))

#Hyper Parameters
host = "localhost"
port = 2000
client = carla.Client(host,port)