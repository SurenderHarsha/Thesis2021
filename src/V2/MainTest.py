#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PPOV9 import *
import sys
#sys.path.append('/home/surender/Downloads/CARLA_0.9.9.4/PythonAPI/carla/dist')
import carla
from carla_real_traffic_scenarios.carla_maps import CarlaMaps
from carla_real_traffic_scenarios.ngsim import NGSimDatasets, DatasetMode
from carla_real_traffic_scenarios.ngsim.scenario import NGSimLaneChangeScenario
#from carla_real_traffic_scenarios.opendd.scenario import OpenDDScenario
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import Scenario

from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from PIL import Image
#from IPython.display import clear_output, Image, display, HTML
import cv2
import math
import pickle
import time
import subprocess
import re
from controller import VehiclePIDController
import torchvision.models as models
import random

#from tensorflow.keras.applications import MobileNet

#mdl = MobileNet(input_shape=(186, 150, 3),include_top=False,weights="imagenet",pooling=max)

model_path = '/data/s4120310/ModelsV9/'
history_path = '/data/s4120310/HistoryV9/'

def prepare_ngsim_scenario(client: carla.Client, data_mode = "train") -> Scenario:
    data_dir = os.environ.get("NGSIM_DIR")
    #data_dir = os.listdir('/home/surender/Downloads/NGSIM')
    assert data_dir, "Path to the directory with NGSIM dataset is required"
    ngsim_map = NGSimDatasets.list()
    ngsim_dataset = ngsim_map[1]
    client.load_world(ngsim_dataset.carla_map.level_path)
    if data_mode == "train":
        return NGSimLaneChangeScenario(
            ngsim_dataset,
            dataset_mode=DatasetMode.TRAIN,
            data_dir=data_dir,
            reward_type=RewardType.DENSE,
            client=client,
        )
    else:
        return NGSimLaneChangeScenario(
            ngsim_dataset,
            dataset_mode=DatasetMode.VALIDATION,
            data_dir=data_dir,
            reward_type=RewardType.DENSE,
            client=client,
        )

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
    
    
    def __init__(self,random_seed = 123456,host = 'localhost',port = 2000):
        self.update_timestep = 1000     # update policy every n timesteps
        self.K_epochs = 200               # update policy for K epochs
        self.eps_clip = 0.2              # clip parameter for PPO
        self.gamma = 0.99                # discount factor
        
        self.lr_actor = 0.0001      # learning rate for actor network
        self.lr_critic = 0.001       # learning rate for critic network
        
        self.save_every = 100
        self.init_val_score = 0.0
        
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        #np.random.seed()
        self.host = host
        self.port = port
        self.scenario = None
        #self.mobilenet = models.mobilenet_v2(pretrained=True).to(device)
        self.mn=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.load_carla()
        self.init_algorithm(1)
        
        
        
        
    def cmd_carla(self):
        os.system("singularity exec --nv /data/s4120310/Singularity.sif /bin/bash /home/carla/CarlaUE4.sh -opengl")
    
    def load_carla(self):
        try:
                
                l = subprocess.check_output("ps -ef | grep Carla | awk '{print $2}'",shell = True)
                s = re.findall(r'\d+', str(l))
                for i in range(len(s)-1):
                    try:
                        os.system('kill -9 '+s[i])
                    except:
                        pass
                print("Killing old carla")
                self.carla_simulator = threading.Thread(target = self.cmd_carla)
                self.carla_simulator.start()
               
                print("Starting carla, loading......")
                time.sleep(20)
                
                self.client = carla.Client(self.host,self.port)
                #time.sleep(5)
                if self.scenario != None:
                    self.scenario.close()
                self.scenario = prepare_ngsim_scenario(self.client)
                self.world = self.client.get_world()
                self.spectator = self.world.get_spectator()
                self.ego_vehicle = prepare_ego_vehicle(self.world)
                self.birdview_producer = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=186, height=150),
                pixels_per_meter=4
                #crop_type=BirdViewCropType.FRONT_AREA_ONLY
                )
        except Exception as e:
                print("Crash failed! reloading until fixed",e)
                self.load_carla()
                #sys.exit(0)
    def init_algorithm(self,out_dim):
        self.policy = PPO(1,out_dim,self.lr_actor,self.lr_critic,self.gamma,self.K_epochs,self.eps_clip,True)
    def load_policy(self,load_file):
        self.policy.load(load_file)
    
    def way_cal(self,ego_vehicle,val,stabilize = False, actual_waypoint = None):
        way = ego_vehicle.get_transform()
        if (val == 0  or val ==1):
            way.location.x += 100        
        if (val == 2 or val == 5):
            way.location.x += 100
            way.location.y += 7
        if (val == 3 or val == 4):
            way.location.x += 100
            way.location.y -= 7
        return way
    def check_safety(self,way):
        d = self.scenario._ngsim_vehicles_in_carla._vehicle_by_vehicle_id
        y_left = way.location.y - 3.5
        y_right = way.location.y + 3.5
        x_lower = self.ego_vehicle.get_location().x + 5
        x_upper = self.ego_vehicle.get_location().x + 25
        car_in_zone = False
        for i in d:
            if d[i].get_location().x > x_lower and d[i].get_location().x < x_upper and d[i].get_location().y > y_left and d[i].get_location().y < y_right:
                car_in_zone = True
        safe = not car_in_zone
        return safe

    def train(self,save_path,hist_path,iteration ,epochs = 500,batch_size = 200,freq_decrease = 1000):
        total_reward_list = []
        epoch_list = []
        step_list = []
        self.scenario.reset(self.ego_vehicle)
        self.world.tick()        
        #torch.cuda.empty_cache()
        #val_score = self.validate()
        #History = [0,0,0,val_score,0,0]
        #f = open(hist_path + "History_"+str(0)+"_"+str(iteration)+".pkl",'wb')
        #pickle.dump(History,f)
        #f.close()
        freq = batch_size
        freq_n = freq_decrease
        min_r_avg = -1
        section = 6
        victory_buffer = 0        
        for epoch in range(epochs):
            self.epoch = epoch
            #torch.cuda.empty_cache()
            step = 0
            try:
                ids = list(self.world.get_actors())[0].id
                carla.command.DestroyActor(ids)
                #del self.ego_vehicle
                #self.scenario = prepare_ngsim_scenario(self.client)
                self.world = self.client.get_world()
                #self.spectator = self.world.get_spectator()
                self.ego_vehicle = prepare_ego_vehicle(self.world)
                #self.birdview_producer = BirdViewProducer(
                #self.client,  # carla.Client
                #target_size=PixelDimensions(width=150, height=186),
                #pixels_per_meter=4,
                #crop_type=BirdViewCropType.FRONT_AREA_ONLY
                #)
                self.scenario.reset(self.ego_vehicle)
                frame = self.world.tick()
                self.ego_vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=4))
                frame = self.world.tick()
                frame = self.world.tick()
            except Exception as e:
                print("Error at epoch begin:,",e)
                total_reward_list.append(np.mean(total_reward_list))
                epoch_list.append(epoch)
                step_list.append(int(np.mean(step_list)))
                self.load_carla()
                continue
            speed = self.scenario._veh.speed
            done = False
            
            total_r = 0
            val = 0
            #way = self.way_cal(self.ego_vehicle,val)
            way = self.scenario._target_lane_waypoint.transform
            #way = self.ego_vehicle.get_transform()
            t_clip_n = 0.0
            t_clip_p = 1.0
            
            s_clip_n = -1.0
            s_clip_p = 1.0
            cmd_buffer = [0]
            yaw_buffer = [0]
            target_buffer = [False]
            steer = 0
            steer_aug = 0.1
            prep_limit = 10*2
            pid = VehiclePIDController(self.ego_vehicle,0.7)
            lock = 0
            if victory_buffer >= 50000:
                section+=1
                victory_buffer = 0
            while not done:
                speed = self.scenario._veh.speed/1.603
                way = self.scenario._target_lane_waypoint.transform
                way.location.x = self.ego_vehicle.get_location().x + 30
                safe = self.check_safety(way)
                '''
                while True:
                    #print(current_frame,c)
                    if current_frame >= c:
                        #print(current_frame,c)
                        break
                '''
                if step < prep_limit:
                    k = pid.run_step(speed,way)
                    #print(k.throttle)
                    self.ego_vehicle.apply_control(carla.VehicleControl(throttle=k.throttle, steer=np.clip(0.0,-1.0,1.0),brake=k.brake))
                else:
                    if lock == 0:
                        pid = VehiclePIDController(self.ego_vehicle,0.7)
                        lock = 1
                    try:
                        birdview = self.birdview_producer.produce(
                            agent_vehicle=self.ego_vehicle  # carla.Actor (spawned vehicle)
                            )
                        a = birdview[0].reshape(1,186,150)
                        a = np.append(a,birdview[1].reshape(1,186,150),axis=0)
                        a = np.append(a,birdview[2].reshape(1,186,150),axis=0)
                        a = np.append(a,birdview[3].reshape(1,186,150),axis=0)
                        a = np.append(a,birdview[4].reshape(1,186,150),axis=0)
                        #rgb = BirdViewProducer.as_rgb(birdview)/255.
                        '''
                        for i in range(3):
                            rgb[:,:,i] = (rgb[:,:,i] - self.mn[i])/self.std[i]
                        in_data = rgb.reshape(1,3,224,224)
            
                        with torch.no_grad():
                            in_data = torch.FloatTensor(in_data).to(device)
                            in_data = self.mobilenet.features(in_data)
                            in_data = nn.AvgPool2d(7,7)(in_data)
                            in_data = nn.Flatten()(in_data)
                            in_data = in_data.detach().cpu().numpy()
                        '''
                        in_data = a.reshape(1,5,186,150)
                        if val == 0 or val == 1:
                            inps= [0,1,0]
                        if val == 2 or val == 5:
                            inps = [0,0,1]
                        if val == 4 or val ==3:
                            inps = [1,0,0]
                        #inps = np.array(inps)
                        #ap = np.concatenate((in_data[0],inps)).reshape(1,1283)
                        inputs = [in_data,np.array(inps).reshape(1,3)]
                        #in_data = a.reshape(1,5,186,150)
                    except Exception as e:
                        print("Error in birds eye:",e)
                        #self.load_carla()
                        break
        
                    action = self.policy.select_action(inputs)
                    '''
                    if action[1]<0:
                        steer -= steer_aug
                    elif action[1]>0:
                        steer += steer_aug
                    if steer < -0.3:
                        steer = -0.3
                    if steer > 0.3:
                        steer = 0.3
                    '''
                    #steer = action[0] 
                    #speed = action[0]*100 + 40
                    
                    brake = 0
                    throttle = 0
                    steer = 0
                    x = (action[0] + 1)/2
                    x = x*100
                    #y = action[1]*5
                    #y = (action[1] + 1)/2
                    #y = y*120
                    way.location.x =  self.ego_vehicle.get_location().x + x
                    #way.location.y = self.ego_vehicle.get_location().y + y
                    #action[0] = action[0]*2 - 1
                    #speed = y
                    #steer = steer*2 - 1
                    #speed = speed*120
                    #_speed = np.clip(action[0], -1,1)
                    #speed = ((_speed + 1)/2)*50 + 10
                    
                    #pid = VehiclePIDController(self.ego_vehicle)
                    k = pid.run_step(speed,way)
                    throttle = k.throttle
                    brake = k.brake
                    steer = k.steer
                    
                    self.ego_vehicle.apply_control(carla.VehicleControl(throttle=np.clip(throttle, t_clip_n, t_clip_p), steer=np.clip(steer, s_clip_n, s_clip_p),brake=np.clip(brake, 0.0, 1.0)))
                    
                
                try:
                    cmd, reward, done, _ = self.scenario.step(self.ego_vehicle)
                except Exception as e:
                    print(e)
                    if step<prep_limit:
                        pass
                    else:
                        self.policy.buffer.rewards.append(0)
                        self.policy.buffer.is_terminals.append(True)
                    break
                    self.load_carla()
                    step = 0
                    self.scenario.reset(self.ego_vehicle)
                    self.world.tick()
                    ids = list(self.world.get_actors())[0].id
                    carla.command.DestroyActor(ids)
                    #self.scenario = prepare_ngsim_scenario(self.client)
                    self.world = self.client.get_world()
                    #self.spectator = self.world.get_spectator()
                    self.ego_vehicle = prepare_ego_vehicle(self.world)
                    #self.birdview_producer = BirdViewProducer(
                    #self.client,  # carla.Client
                    #target_size=PixelDimensions(width=150, height=186),
                    #pixels_per_meter=4,
                    #crop_type=BirdViewCropType.FRONT_AREA_ONLY
                    #)
                    self.scenario.reset(self.ego_vehicle)
                    #self.scenario.reset(self.ego_vehicle)
                    frame = self.world.tick()
                    self.ego_vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=4))
                    frame = self.world.tick()
                    frame = self.world.tick()
                    done = False
                    
                    total_r = 0
                    val = 0
                    way = scenario._target_lane_waypoint.transform
                    #way = self.ego_vehicle.get_transform()
                    cmd_buffer = [0]
                    yaw_buffer = [0]
                    t_clip_n = 0.0
                    t_clip_p = 1.0
                    steer = 0
                    steer_aug = 0.05
                    prep_limit = 10*2
                    pid = VehiclePIDController(self.ego_vehicle,0.7)
                    lock = 0
                    s_clip_n = -1.0
                    s_clip_p = 1.0
                    continue
                    
                val = cmd.value
                #val = cmd.value
                cmd_buffer.append(val)
                yaw_buffer.append(self.ego_vehicle.get_transform().rotation.yaw)
                target_buffer.append(_['on_target_lane'])
                if section == 2:
                    if len(cmd_buffer) > 10:
                        if sum(cmd_buffer[-10:]) == 0 and _['on_target_lane'] and sum([abs(x) for x in yaw_buffer[-10:]])/10<=10 and total_r>=0.2:
                            reward = 1 
                            done = True
                if section == 3:
                    if len(cmd_buffer) > 10:
                        if sum(cmd_buffer[-10:]) == 0 and _['on_target_lane'] and sum([abs(x) for x in yaw_buffer[-10:]])/10<=10 and total_r>=0.4:
                            reward = 1 
                            done = True
                if section == 4:
                    if len(cmd_buffer) > 10:
                        if sum(cmd_buffer[-10:]) == 0 and _['on_target_lane'] and sum([abs(x) for x in yaw_buffer[-10:]])/10<=10 and total_r>=0.6:
                            reward = 1 
                            done = True
                if section == 5:
                    if len(cmd_buffer) > 10:
                        if sum(cmd_buffer[-10:]) == 0 and _['on_target_lane'] and sum([abs(x) for x in yaw_buffer[-10:]])/10<=10 and total_r>=0.8:
                            reward = 1 
                            done = True
                if section>=6:
                    pass
                
                #way = self.way_cal(self.ego_vehicle,val)
                #reward = np.clip(reward,-1,1)
                if step < prep_limit:
                    pass
                else:
                    #print("HI")
                    #p.buffer.rewards.append(reward)
                    #p.buffer.is_terminals.append(done)
                    self.policy.buffer.rewards.append(reward)
                    self.policy.buffer.is_terminals.append(done)
                if reward >= 1:
                    victory_buffer+=1
                    print("Its a win")
                total_r += reward
                step += 1
                '''
                if step % freq ==0 :
                    #print(step)
                    self.policy.update()
                if step % freq_n == 0:
                    self.policy.decay_action_std(0.01,0.001)
                '''
                frame = self.world.tick()
            
            try:
                '''
                if len(total_reward_list) > 50:
        
                    if sum(total_reward_list[-50:])/len(total_reward_list[-50:]) > min_r_avg or sum(step_list) > freq_n:
                        #freq_n += 1000
                        #decay_c -= 50
                        #print("Decaying:",self.pop.action_std)
                        self.policy.decay_action_std(0.001,0.1)
                        
                        min_r_avg = sum(total_reward_list[-50:])/len(total_reward_list[-50:])
                '''
                if epoch % freq== 0 and epoch != 0:
                        #print(steer,avg_steer,throttle,brake,steer_w)
                        print("Update with batches:",len(self.policy.buffer.states))
                        self.policy.update()
                        
                        #torch.cuda.empty_cache()
                        val_score = self.validate()
                        
                        #self.policy.decay_action_std(0.0005,0.1)
                        print("Saving model and history")
                        History = [epoch_list,total_reward_list,step_list,val_score,section,victory_buffer]
                        total_reward_list = []
                        epoch_list = []
                        step_list = []
                        self.policy.save(save_path + str("Model_"+str(epoch)+"_"+str(iteration)+".mdl"))
                        f = open(hist_path + "History_"+str(epoch)+"_"+str(iteration)+".pkl",'wb')
                        pickle.dump(History,f)
                        f.close()
                        
            except Exception as e:
                print("Error in update:",e)
                pass
            
            #cv2.destroyAllWindows()
            if total_r >=0.9:
                print("Win Episode!")
            print("Epoch:",epoch, "Total Reward:",total_r,"Number of Steps:",step)
            if total_r == 0  and step == 0:
                self.load_carla()
            total_reward_list.append(total_r)
            epoch_list.append(epoch)
            step_list.append(step)
    def validate(self):
        val_success = []
        try:
            self.scenario.close()
            #self.scenario_val.close()
        except:
            pass
        
        self.scenario = prepare_ngsim_scenario(self.client,"Val")
        self.world = self.client.get_world()
        self.ego_vehicle = prepare_ego_vehicle(self.world)
        self.birdview_producer = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=186, height=150),
                pixels_per_meter=4
                #crop_type=BirdViewCropType.FRONT_AREA_ONLY
        )
        '''
        except Exception as e:
            print("Init Val er:",e)
            self.load_carla()
            try:
                self.scenario.close()
            except:
                pass
            self.scenario = prepare_ngsim_scenario(self.client,"Val")
            self.world = self.client.get_world()
            self.ego_vehicle = prepare_ego_vehicle(self.world)
            self.birdview_producer = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=150, height=186),
                pixels_per_meter=4,
                crop_type=BirdViewCropType.FRONT_AREA_ONLY
            )
        '''
        #self.scenario_val.reset(self.ego_vehicle)
        self.scenario.reset(self.ego_vehicle)
        self.world.tick()
        val_count = 0
        for i in range(50):
            t_clip_n = 0.0
            t_clip_p = 1.0
        
            s_clip_n = -1.0
            s_clip_p = 1.0
            #torch.cuda.empty_cache()
            step = 0
            #del self.ego_vehicle
            #self.scenario = prepare_ngsim_scenario(self.client,"Val")
            ids = list(self.world.get_actors())[0].id
            carla.command.DestroyActor(ids)
            self.world = self.client.get_world()
            #self.spectator = self.world.get_spectator()
            self.ego_vehicle = prepare_ego_vehicle(self.world)
            #self.birdview_producer = BirdViewProducer(
            #    self.client,  # carla.Client
            #    target_size=PixelDimensions(width=150, height=186),
            #    pixels_per_meter=4,
            #    crop_type=BirdViewCropType.FRONT_AREA_ONLY
            #    )
            self.scenario.reset(self.ego_vehicle)
            c = self.world.tick()
            self.ego_vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=4))
            c = self.world.tick()
            c = self.world.tick()
            speed = self.scenario._veh.speed
            
            '''
            except Exception as e:
                print("Error in val init:",e)
                val_success.append(0)
                self.load_carla()
                continue
            '''
            
            done = False
            reward = 0
            val = 0
            #way = self.way_cal(self.ego_vehicle,val)
            way = self.scenario._target_lane_waypoint.transform
            cmd_buffer = [0]
            yaw_buffer = [0]
            total_rew = 0
            target_buffer = [False]
            steer = 0
            steer_aug = 0.05
            prep_limit = 10*2
            pid = VehiclePIDController(self.ego_vehicle,0.7)
            lock = 0
            fin = 0
            while not done:
                    speed = self.scenario._veh.speed/1.603
                    way = self.scenario._target_lane_waypoint.transform
                    way.location.x = self.ego_vehicle.get_location().x + 30
                    safe = self.check_safety(way)
                    '''
                    while True:
                        #print(current_frame,c)
                        if current_frame >= c:
                            #print(current_frame,c)
                            break
                    '''
                    if step < prep_limit:
                        k = pid.run_step(speed,way)
                        #print(k.throttle)
                        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=k.throttle, steer=np.clip(0.0,-1.0,1.0),brake=k.brake))
                    else:
                        if lock == 0:
                            pid = VehiclePIDController(self.ego_vehicle,0.7)
                            lock = 1
                        birdview = self.birdview_producer.produce(
                            agent_vehicle=self.ego_vehicle  # carla.Actor (spawned vehicle)
                            )
                        a = birdview[0].reshape(1,186,150)
                        a = np.append(a,birdview[1].reshape(1,186,150),axis=0)
                        a = np.append(a,birdview[2].reshape(1,186,150),axis=0)
                        a = np.append(a,birdview[3].reshape(1,186,150),axis=0)
                        a = np.append(a,birdview[4].reshape(1,186,150),axis=0)
                        #a = np.append(a,birdview[4].reshape(1,224,224),axis=0)
                        '''
                        rgb = BirdViewProducer.as_rgb(birdview)/255.
                        for i in range(3):
                            rgb[:,:,i] = (rgb[:,:,i] - self.mn[i])/self.std[i]
                        #in_data = a.reshape(1,5,186,150)
                        in_data = rgb.reshape(1,3,224,224)
            
                        with torch.no_grad():
                            in_data = torch.FloatTensor(in_data).to(device)
                            in_data = self.mobilenet.features(in_data)
                            in_data = nn.AvgPool2d(7,7)(in_data)
                            in_data = nn.Flatten()(in_data)
                            in_data = in_data.detach().cpu().numpy()
                        '''
                        in_data = a.reshape(1,5,186,150)
                        if val == 0 or val == 1:
                            inps= [0,1,0]
                        if val == 2 or val == 5:
                            inps = [0,0,1]
                        if val == 4 or val ==3:
                            inps = [1,0,0]
                        #inps = np.array(inps)
                        #ap = np.concatenate((in_data[0],inps)).reshape(1,1283)
                        inputs = [in_data,np.array(inps).reshape(1,3)]
                        
                        with torch.no_grad():
                            action = self.policy.select_action(inputs,True)
                        #action = action * 2 - 1
                        steer = 0
                        x = (action[0] + 1)/2
                        x = x*100
                        #y = action[1]*5
                        #y = (action[1] +1)/2
                        #y = y*120
                        way.location.x =  self.ego_vehicle.get_location().x + x
                        #way.location.y = self.ego_vehicle.get_location().y + y
                        #speed = y
                        brake = 0
                        throttle = 0
                        #_speed = np.clip(action[0], -1,1)
                        #speed = ((_speed + 1)/2)*50 + 10
            
            
            
                        #pid = VehiclePIDController(self.ego_vehicle)
                        k = pid.run_step(speed,way)
                        throttle = k.throttle
                        brake = k.brake
            
                        avg_steer = k.steer
                        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=np.clip(avg_steer, s_clip_n, s_clip_p),brake=brake))
            
                    #ego_vehicle.apply_control(carla.VehicleControl(throttle=np.clip(throttle, t_clip_n, t_clip_p), steer=np.clip(action[1], s_clip_n, s_clip_p)))#,brake=np.clip(brake, 0.0, 1.0)))
        
        
                    try:
                        
                        cmd, reward, done, _ = self.scenario.step(self.ego_vehicle)
                    except Exception as e:
                        print(e)
                        break
                    '''
                    except:
                        break
                    
                    if reward < 0 :
                        reward = -0.1
                    '''
                    '''
                    if reward <= -1:
                        reward = 0
                    '''
        
                    #print(reward, cmd)
                    total_rew += reward
                    val = cmd.value
                    cmd_buffer.append(val)
                    yaw_buffer.append(self.ego_vehicle.get_transform().rotation.yaw)
                    target_buffer.append(_['on_target_lane'])
                    #fin = 0
                    if len(cmd_buffer) > 10:
                        if sum(cmd_buffer[-10:]) == 0 and _['on_target_lane'] and sum([abs(x) for x in yaw_buffer[-10:]])/10<=10 and total_rew>=0.6:
                            #reward = 1
                            #done = True
                            #fin = 1
                            pass
                    '''
                    if len(cmd_buffer) > 10:
                        if _['on_target_lane'] and all(item == True for item in target_buffer[-10:]):
                            reward = 1 + random.random()
                            done = True
        
                    '''
                    
                    #total_rew += reward
                    #way = self.way_cal(self.ego_vehicle,val)
                    #way = scenario._target_lane_waypoint.transform 
                    c = self.world.tick()
                    step+=1
            if step>1:
                val_count+=1
            if total_rew >0.0:
                print("Success!")
                #val_reward.append(reward)
                #val_count += 1
                if fin == 1:
                    total_rew += 1.0
                val_success.append(1)
            else:
                #val_reward.append(reward)
                val_success.append(0)
            #print(total_rew)
        print(sum(val_success)," <-- Number of success")
        try:
            self.scenario.close()
        except:
            pass
        self.scenario = prepare_ngsim_scenario(self.client)
        self.world = self.client.get_world()
        #self.spectator = self.world.get_spectator()
        self.ego_vehicle = prepare_ego_vehicle(self.world)
        self.birdview_producer = BirdViewProducer(
            self.client,  # carla.Client
            target_size=PixelDimensions(width=186, height=150),
            pixels_per_meter=4
            #crop_type=BirdViewCropType.FRONT_AREA_ONLY
        )
        
        '''
        except Exception as e:
            print("Re init Train:",e)
            self.load_carla()
            self.scenario = prepare_ngsim_scenario(self.client)
            self.world = self.client.get_world()
            #self.spectator = self.world.get_spectator()
            self.ego_vehicle = prepare_ego_vehicle(self.world)
            self.birdview_producer = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=150, height=186),
                pixels_per_meter=4,
                crop_type=BirdViewCropType.FRONT_AREA_ONLY
            )
        '''
        self.scenario.reset(self.ego_vehicle)
        self.world.tick() 
        if val_count == 0:
            val_count = 1
        val_scores = sum(val_success)/len(val_success)
        return val_scores
if __name__ == "__main__":
    args = sys.argv
    c = CarlaManager()
    batch_size = int(args[3])
    try:
        load_file_name = args[4]
        
        load_q = bool(args[5])
    except:
        load_file_name = ""
        load_q = False
    
    if load_q:
        c.load_policy(load_file_name)
    
    total_epochs = int(args[1])
    iteration = int(args[2])
    
    
    c.train(model_path,history_path,iteration,total_epochs,batch_size)
    print("Done Training!")
            



        
        
        
        
