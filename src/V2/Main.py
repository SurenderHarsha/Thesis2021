#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PPO import *
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



model_path = '/data/s4120310/Models/'
history_path = '/data/s4120310/History/'

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
        reward_type=RewardType.DENSE,
        client=client,
    )

'''
def prepare_opendd_scenario(client: carla.Client) -> Scenario:
    data_dir = os.environ.get("OPENDD_DIR")
    assert data_dir, "Path to the directory with openDD dataset is required"
    maps = ["rdb1", "rdb2", "rdb3", "rdb4", "rdb5", "rdb6", "rdb7"]
    map_name = random.choice(maps)
    carla_map = getattr(CarlaMaps, map_name.upper())
    client.load_world(carla_map.level_path)
    return OpenDDScenario(
        client,
        dataset_dir=data_dir,
        dataset_mode=DatasetMode.TRAIN,
        reward_type=RewardType.DENSE,
        place_name=map_name,
    )

'''
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
    
    
    def __init__(self,random_seed = 663073192825619597,host = 'localhost',port = 2000):
        self.update_timestep = 1000     # update policy every n timesteps
        self.K_epochs = 40               # update policy for K epochs
        self.eps_clip = 0.2              # clip parameter for PPO
        self.gamma = 0.99                # discount factor
        
        self.lr_actor = 0.0003       # learning rate for actor network
        self.lr_critic = 0.001       # learning rate for critic network
        
        self.save_every = 500
        
        
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        #np.random.seed()
        self.host = host
        self.port = port
        self.scenario = None
        
        self.load_carla()
        self.init_algorithm(2)
        
        
        
        
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
                if self.scenario != None:
                    self.scenario.close()
                self.scenario = prepare_ngsim_scenario(self.client)
                self.world = self.client.get_world()
                self.spectator = self.world.get_spectator()
                self.ego_vehicle = prepare_ego_vehicle(self.world)
                self.birdview_producer = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=150, height=186),
                pixels_per_meter=4,
                crop_type=BirdViewCropType.FRONT_AREA_ONLY
                )
        except Exception as e:
                print("Crash failed! reloading until fixed",e)
                self.load_carla()
                #sys.exit(0)
    def init_algorithm(self,out_dim):
        self.policy = PPO(1,out_dim,self.lr_actor,self.lr_critic,self.gamma,self.K_epochs,self.eps_clip,True)
    def load_policy(self,load_file):
        self.policy.load(load_file)
    
    def train(self,save_path,hist_path,iteration ,epochs = 500,batch_size = 5,freq_decrease = 5):
        total_reward_list = []
        epoch_list = []
        step_list = []
        
        torch.cuda.empty_cache()
        
        freq = batch_size
        freq_n = freq_decrease
        for epoch in range(epochs):
            
            step = 0
            try:
                self.scenario.reset(self.ego_vehicle)
                frame = self.world.tick()
            except:
                total_reward_list.append(np.mean(total_reward_list))
                epoch_list.append(epoch)
                step_list.append(int(np.mean(step_list)))
                self.load_carla()
                continue
            done = False
            
            total_r = 0
            val = 0
            
            
            t_clip_n = 0.0
            t_clip_p = 1.0
            
            s_clip_n = -1.0
            s_clip_p = 1.0
            
            
            while not done:
                '''
                while True:
                    #print(current_frame,c)
                    if current_frame >= c:
                        #print(current_frame,c)
                        break
                '''
                try:
                    birdview = self.birdview_producer.produce(
                        agent_vehicle=self.ego_vehicle  # carla.Actor (spawned vehicle)
                        )
                except:
                    print(e)
                    self.load_carla()
                    step = 0
                    self.scenario.reset(self.ego_vehicle)
                    frame = self.world.tick()
                    done = False

                    total_r = 0
                    val = 0


                    t_clip_n = 0.0
                    t_clip_p = 1.0

                    s_clip_n = -1.0
                    s_clip_p = 1.0
                    continue
    
                in_data = birdview[:5,:,:]
                in_data = in_data.reshape((1,5,186,150))
                #in_data = input_data.reshape((1,3,320,320))
                action = self.policy.select_action(in_data)
                #print(action)
                '''
                if (val == 0  or val ==1):
                    s_clip_n = -0.15
                    s_clip_p = 0.15
                    t_clip_n = 0.4
                    t_clip_p = 1.0
                
                if (val == 2 or val == 5):
                    s_clip_n = 0.25
                    s_clip_p = 0.8
                    t_clip_n = 0.0
                    t_clip_p = 0.4
                
                if (val == 3 or val == 4):
                    s_clip_n = -0.8
                    s_clip_p = -0.25
                    t_clip_n = 0.0
                    t_clip_p = 0.4
                '''
                    
                
                t_clip_n = 0.0
                t_clip_p = 1.0
        
                s_clip_n = -1.0
                s_clip_p = 1.0    
                
                brake = 0.0
                throttle = 0.0
                
                if action[0] <0:
                    brake = action[0]
                    throttle = 0.0
                else:
                    throttle = action[0]
                    brake = 0.0
                '''
                if (val == 0  or val ==1):
                    s_clip_n = -0.15
                    s_clip_p = 0.15
                    t_clip_n = 0.3
                    t_clip_p = 1.0
                
                if (val == 2 or val == 5):
                    s_clip_n = 0.25
                    s_clip_p = 0.8
                    t_clip_n = 0.0
                    t_clip_p = 0.4
                
                if (val == 3 or val == 4):
                    s_clip_n = -0.8
                    s_clip_p = -0.25
                    t_clip_n = 0.0
                    t_clip_p = 0.4
                '''
                #if epoch < 20:
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=np.clip(throttle, t_clip_n, t_clip_p), steer=np.clip(action[1], s_clip_n, s_clip_p),brake=np.clip(brake, 0.0, 1.0)))
                
                
                try:
                    cmd, reward, done, _ = self.scenario.step(self.ego_vehicle)
                except Exception as e:
                    print(e)
                    self.load_carla()
                    step = 0
                    self.scenario.reset(self.ego_vehicle)
                    frame = self.world.tick()
                    done = False
                    
                    total_r = 0
                    val = 0
                    
                    
                    t_clip_n = 0.0
                    t_clip_p = 1.0
                    
                    s_clip_n = -1.0
                    s_clip_p = 1.0
                    continue
                    
                val = cmd.value
                #print(done)
                #if done:
                #    print(_)
                #print(_)
                
                
                #v = self.ego_vehicle.get_velocity()
                #kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
                
                '''
                if kmh < 60 & kmh > 0.2:
                    #done = False
                    reward += 1 #-1
                    # Reward lighter steering when moving
                    if np.abs(action[1]) < 0.3:
                        reward += 1
                    elif np.abs(action[1]) > 0.5 and np.abs(action[1]) < 0.9:
                        reward -= 0.1
                    elif np.abs(action[1]) >= 0.9:
                        reward -= 0.2
                elif kmh < 0.2:
                    reward -= 0.1
                else:
                    #print("Maybe never")
                    reward += 0.01
                    if np.abs(action[1]) < 0.3:
                        reward += 0.12
                    # Reduce score for heavy steering
                    if np.abs(action[1]) > 0.5 and np.abs(action[1]) < 0.9:
                        reward -= 0.17
                    elif np.abs(action[1]) >= 0.9:
                        reward -= 0.21
                '''
                if (val == 0  or val ==1):
                    if throttle > 0.2 and throttle < 0.8:
                        reward += 0.2
                
                if (val == 2 or val == 5):
                    if action[1] > 0.2 and action[1] <0.5:
                        reward += 0.2
                
                if (val == 3 or val == 4):
                    if action[1] <-0.2 and action[1] > -0.5:
                        reward += 0.19
                '''
                rgb = BirdViewProducer.as_rgb(birdview)
                cv2.imshow('Frame',rgb)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                '''
                self.policy.buffer.rewards.append(reward)
                self.policy.buffer.is_terminals.append(done)
                
                total_r += reward
                step += 1
                
                if step % freq ==0 :
                    #print(step)
                    self.policy.update()
                if step % freq_n == 0:
                    self.policy.decay_action_std(0.01,0.001)
                frame = self.world.tick()
            
            try:
                if step > 1 and len(self.policy.buffer.states) > 1:   
                    self.policy.update()
                else:
                    pass
            except Exception as e:
                print("Error:",e)
                pass
            
            #cv2.destroyAllWindows()
            print("Epoch:",epoch, "Total Reward:",total_r,"Number of Steps:",step)
            total_reward_list.append(total_r)
            epoch_list.append(epoch)
            step_list.append(step)
            
            if epoch % self.save_every == 0:
                print("Saving model and history")
                History = [epoch_list,total_reward_list,step_list]
                total_reward_list = []
                epoch_list = []
                step_list = []
                self.policy.save(save_path + str("Model_"+str(epoch)+"_"+str(iteration)+".mdl"))
                f = open(hist_path + "History_"+str(epoch)+"_"+str(iteration)+".pkl",'wb')
                pickle.dump(History,f)
                f.close()


if __name__ == "__main__":
    args = sys.argv
    c = CarlaManager()
    try:
        load_file_name = args[3]
        
        load_q = bool(args[4])
    except:
        load_file_name = ""
        load_q = False
    
    if load_q:
        c.load_policy(load_file_name)
    
    total_epochs = int(args[1])
    iteration = int(args[2])
    
    
    c.train(model_path,history_path,iteration,total_epochs)
    print("Done Training!")
            



        
        
        
        
