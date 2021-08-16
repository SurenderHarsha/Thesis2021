#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import carla
import numpy as np
from PIL import Image
import time

import warnings
warnings.filterwarnings("ignore")

class VehicleManager(object):
    
    
    def __init__(self,ego_vehicle,world,scenario,baseline = False,neighbours = 4):
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.scenario = scenario
        self.baseline = baseline
        
        self.ego_front_buffer = None
        self.ego_back_buffer = None
        self.ego_gnss_buffer = None
        
        self.current_frame = None
        self.previous_frame = None
        self.rgb_fov = 100
        self.neighbours = neighbours
        
        self.neighbour_front_image_data = []
        self.neighbour_back_image_data = []
        self.neighbour_gnss_data = []
        
        self.input_h = 300
        self.input_w = 300
        
        self.neighbour_profiles = []
        try:
            
            self.vehicles = self.scenario._ngsim_vehicles_in_carla._vehicle_by_vehicle_id
        except:
            print("Exception! The scenario is not being reset properly!")
            self.scenario.reset(self.ego_vehicle)
            self.vehicles = self.scenario._ngsim_vehicles_in_carla._vehicle_by_vehicle_id
        self.vehicleSensorslist = {}
        
        #Initialize vehicle sensors
        self.init_sensors()
        
        
    
    def init_sensors(self):
        self.ego_setup_sensors()
        
        for vehicle in self.vehicles:
            self.vehicleSensorslist[vehicle] = VehicleSensors(self.vehicles[vehicle],self.world)
            
        for vehicle in self.vehicleSensorslist:
            self.vehicleSensorslist[vehicle].init_rgb_back_camera()
            self.vehicleSensorslist[vehicle].init_rgb_front_camera()
            self.vehicleSensorslist[vehicle].init_gnss()
            self.vehicleSensorslist[vehicle].stop_sensors()
        
        
    def stop_all_sensors(self):
        for vehicle in self.vehicleSensorslist:
            self.vehicleSensorslist[vehicle].stop_sensors()
    
    def update_neighbours(self):
        self.stop_all_sensors()
        
        dist = {}
        self.neighbour_profiles = []
        for vehicle in self.vehicles:
            d = self.ego_vehicle.get_location().distance(self.vehicles[vehicle].get_location())
            dist[vehicle] = d
        updated_dist = dict(sorted(dist.items(), key=lambda item: item[1]))
        self.neighbour_profiles = list(updated_dist.keys())[:self.neighbours]
        for neighbour in self.neighbour_profiles:
            self.vehicleSensorslist[neighbour].start_sensors()
        
    def update_buffers(self):
        self.neighbour_front_image_data = []
        self.neighbour_back_image_data = []
        self.neighbour_gnss_data = []
        
        for neighbour in self.neighbour_profiles:
            if self.current_frame != self.vehicleSensorslist[neighbour].gnss_frame:
                print("Frame Mismatch!")
                time.sleep(0.1)
            self.neighbour_back_image_data.append(self.vehicleSensorslist[neighbour].back_image_buffer)
            self.neighbour_front_image_data.append(self.vehicleSensorslist[neighbour].front_image_buffer)
            self.neighbour_gnss_data.append(self.vehicleSensorslist[neighbour].gnss_buffer)
        
        
    def return_data(self):
        ego_data = [self.ego_front_buffer,self.ego_back_buffer,self.ego_gnss_buffer]
        neighbour_profiles = [self.neighbour_front_image_data,self.neighbour_back_image_data,self.neighbour_gnss_data]
        return ego_data,neighbour_profiles
    
        
        
    def set_gnss(self,data):
        self.current_frame = data.frame
        
        input_data = np.array([data.altitude,data.latitude, data.longitude])
        self.ego_gnss_buffer = input_data
        
    
    def set_front_buffer(self,img):
        #self.front_frame = img.frame
        array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8")) 
        array = np.reshape(array, (img.height, img.width, 4)) # RGBA format
        array = array[:, :, :3] #  Take only RGB
        img = Image.fromarray(array)
        img = img.resize((self.input_h, self.input_w), Image.ANTIALIAS)
        input_data = np.array(img)
        self.ego_front_buffer = input_data
        
    def set_back_buffer(self,img):
        #self.back_frame = img.frame
        array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8")) 
        array = np.reshape(array, (img.height, img.width, 4)) # RGBA format
        array = array[:, :, :3] #  Take only RGB
        img = Image.fromarray(array)
        img = img.resize((self.input_h, self.input_w), Image.ANTIALIAS)
        input_data = np.array(img)
        self.ego_back_buffer = input_data
    
    def ego_setup_sensors(self):
        geo_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        geo_loc = carla.Location(0,0,0)
        geo_rot = carla.Rotation(0,0,0)
        geo_transform = carla.Transform(geo_loc,geo_rot)
        self.ego_gnss = self.world.spawn_actor(geo_bp,geo_transform, attach_to=self.ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.ego_gnss.listen(lambda dat: self.set_gnss(dat))
        
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(320))
        cam_bp.set_attribute("image_size_y",str(320))
        cam_bp.set_attribute("fov",str(self.rgb_fov))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        self.ego_front_cam = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        #self.rgb_front_listener = ego_cam
        self.ego_front_cam.listen(lambda image: self.set_front_buffer(image))
        
        bcam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bcam_bp.set_attribute("image_size_x",str(320))
        bcam_bp.set_attribute("image_size_y",str(320))
        bcam_bp.set_attribute("fov",str(self.rgb_fov))
        bcam_location = carla.Location(-2,0,1)
        bcam_rotation = carla.Rotation(0,180,0)
        bcam_transform = carla.Transform(bcam_location,bcam_rotation)
        self.ego_back_cam = self.world.spawn_actor(bcam_bp,bcam_transform,attach_to=self.ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        #self.rgb_back_listener = ego_cam
        self.ego_back_cam.listen(lambda image: self.set_back_buffer(image))


class VehicleSensors(object):
    
    
    def __init__(self,vehicle_actor,world):
        
        self.actor_object = vehicle_actor
        self.vehicle_id = vehicle_actor.id
        self.world = world
        
        self.radar_range = 10.0
        self.radar_listener = None
        
        self.rgb_fov = 100
        self.rgb_front_listener = None
        self.rgb_back_listener = None
        
        self.gnss_listener = None
        
        self.current_frame = None
        
        self.front_image_buffer = None
        self.back_image_buffer = None
        self.gnss_buffer = None
        
        self.front_frame = None
        self.back_frame = None
        self.gnss_frame = None
        
        self.p_front_frame = -1
        self.p_back_frame = -1
        self.p_gnss_frame = -1
        
        self.input_h = 300
        self.input_w = 300
        
    '''   
    def init_radar(self):
        ldr_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        ldr_loc = carla.Location(0,0,0)
        ldr_rot = carla.Rotation(0,0,0)
        ldr_bp.set_attribute("range",str(self.radar_range))
        ldr_transform = carla.Transform(ldr_loc,ldr_rot)
        ego_ldr = world.spawn_actor(ldr_bp,ldr_transform, attach_to=self.actor_object, attachment_type=carla.AttachmentType.Rigid)
        self.radar_listener = ego_ldr
        self.radar_listener.listen(lambda dat: load_radar(dat))
    '''
    
    
    
    def start_sensors(self):
        self.rgb_back_listener.listen(lambda image: self.set_back_buffer(image))
        self.rgb_front_listener.listen(lambda image: self.set_front_buffer(image))
        self.gnss_listener.listen(lambda dat: self.set_gnss(dat))
        
        
    def stop_sensors(self):
        if self.rgb_back_listener.is_listening:
            self.rgb_back_listener.stop()
        if self.rgb_front_listener.is_listening:
            self.rgb_front_listener.stop()
        if self.gnss_listener.is_listening:
            self.gnss_listener.stop()
    
    def set_front_buffer(self,img):
        self.front_frame = img.frame
        array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8")) 
        array = np.reshape(array, (img.height, img.width, 4)) # RGBA format
        array = array[:, :, :3] #  Take only RGB
        img = Image.fromarray(array)
        img = img.resize((self.input_h, self.input_w), Image.ANTIALIAS)
        input_data = np.array(img)
        self.front_image_buffer = input_data
        
    def set_back_buffer(self,img):
        self.back_frame = img.frame
        array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8")) 
        array = np.reshape(array, (img.height, img.width, 4)) # RGBA format
        array = array[:, :, :3] #  Take only RGB
        img = Image.fromarray(array)
        img = img.resize((self.input_h, self.input_w), Image.ANTIALIAS)
        input_data = np.array(img)
        self.back_image_buffer = input_data
        
    def set_gnss(self,data):
        self.gnss_frame = data.frame
        
        input_data = np.array([data.altitude,data.latitude, data.longitude])
        self.gnss_buffer = input_data
    
    def init_rgb_front_camera(self):
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(320))
        cam_bp.set_attribute("image_size_y",str(320))
        cam_bp.set_attribute("fov",str(self.rgb_fov))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_cam = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.actor_object, attachment_type=carla.AttachmentType.Rigid)
        self.rgb_front_listener = ego_cam
        self.rgb_front_listener.listen(lambda image: self.set_front_buffer(image))
        
    
    def init_rgb_back_camera(self):
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(320))
        cam_bp.set_attribute("image_size_y",str(320))
        cam_bp.set_attribute("fov",str(self.rgb_fov))
        cam_location = carla.Location(-2,0,1)
        cam_rotation = carla.Rotation(0,180,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_cam = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.actor_object, attachment_type=carla.AttachmentType.Rigid)
        self.rgb_back_listener = ego_cam
        self.rgb_back_listener.listen(lambda image: self.set_back_buffer(image))
    
    def init_gnss(self):
        sadr_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        sadr_loc = carla.Location(0,0,0)
        sadr_rot = carla.Rotation(0,0,0)
        #ldr_bp.set_attribute("range",'10.0')
        sadr_transform = carla.Transform(sadr_loc,sadr_rot)
        ego_sadr = self.world.spawn_actor(sadr_bp,sadr_transform, attach_to=self.actor_object, attachment_type=carla.AttachmentType.Rigid)
        self.gnss_listener = ego_sadr
        self.gnss_listener.listen(lambda dat: self.set_gnss(dat))
    
     