import gym
import numpy as np
import sys
sys.path.append('CityFlow/dist/CityFlow-0.1.1-py3.7-linux-x86_64.egg')
import cityflow
import json
import random
class MyEnv(gym.Env):
    def __init__(self, eng_config_file, max_step = 3600):
        f = open(eng_config_file, 'r')
        roadnet_file = json.load(f)['roadnetFile']
        f = open(roadnet_file, 'r')
        a = json.load(f)
        f.close()
        self.eng_config_file = eng_config_file
        self.eng = cityflow.Engine(self.eng_config_file, thread_num=1)
        self.eng.set_save_replay(True)
        self.reward_flag = 1
        self.pre_lane_vehicle = {}
        self.now_step = 0
        self.max_step = max_step
        self.action_space = gym.spaces.Discrete(8)  # 动作空间
        self.seconds_per_step = 10
        self.redtime = 5
        self.info_enable = 0
        self.intersections = {}
        self.agents = {}
        self.single_phase_agents = {}
        self.observation_features = ['lane_vehicle_num', 'waiting_vehicle_num']
        self.inverse_road = {}
        self.vehicle_start_time = {}
        self.last_action = None
        for o in a['roads']:
            key1 = o['startIntersection'] + o['endIntersection']
            key2 = o['endIntersection'] + o['startIntersection']
            self.inverse_road[key1] = []
            self.inverse_road[key2] = []
        for o in a['roads']:
            key1 = o['startIntersection'] + o['endIntersection']
            key2 = o['endIntersection'] + o['startIntersection']
            self.inverse_road[key1].append(o)
            self.inverse_road[key2].append(o)
        self.tmp = {}

        for o in self.inverse_road.values():
            if len(o) == 2:
                self.tmp[o[0]['id']] = o[1]['id']
                self.tmp[o[1]['id']] = o[0]['id']
        self.inverse_road = self.tmp
        for o in a['intersections']:
            agent = 0
            if len(o['trafficLight']['lightphases']) == 9:
                #links = [l for o['trafficLight']['lightphases']]
                if len(set([d for p in o['trafficLight']['lightphases'] for d in p['availableRoadLinks']])) != max([len(p['availableRoadLinks']) for p in o['trafficLight']['lightphases']]):
                    agent = 1
                    self.agents[o['id']] = o['roads']
                else:
                    self.single_phase_agents[o['id']] = o

            self.intersections[o['id']] = o
            self.intersections[o['id']]['have_signal'] = agent
        self.agentlist = list(self.agents.keys())
        self.roads = {}
        for o in a['roads']:

            self.roads[o['id']] = {
                'start_inter': o['startIntersection'],
                'end_inter': o['endIntersection'],
                'length': ((o['points'][0]['x']-o['points'][1]['x'])**2+(o['points'][0]['y']-o['points'][1]['y'])**2)**0.5,
                'speed_limit': float(o['lanes'][0]['maxSpeed']),
                'num_lanes': len(o['lanes']),
                # 'angle':o['_compass_angle']
            }


        self.observation_space = gym.spaces.Box(low=-1e10, high=1e10, shape=(len(self.agents),32,))
        self.action_space = gym.spaces.Discrete(8)
		# 其他成员

        self.now_phases = np.full((len(self.agents)),-1)#{o:-1 for o in self.agents}
        for agent_id, roads in self.agents.items():
            for road in roads:
                for i in range(3):
                    lane = road + '_' + str(i)
                    self.pre_lane_vehicle[lane] = set()
        
        self.agent_lane = [[str(road) + '_' + str(i) for i in range(2) for road in agentroads[:4]] for agentroads in self.agents.values()]
        
        self.lane_to_agenti = {}
        for k,v in self.roads.items():
            if v['end_inter'] not in self.agentlist:
                continue
            for i in range(3):
                self.lane_to_agenti[f"{k}_{i}"] = self.agentlist.index(v['end_inter'])
                
        self.reset()
    
    def _get_roadmask(self):
        return

    def _get_info(self):
        info = {}
        if(self.info_enable == 0):
            return info
        else:
            v_list = self.eng.get_vehicles()
            for vehicle in v_list:
                info[vehicle] = self.eng.get_vehicle_info(vehicle)
            return info
        
    def _get_pass(self, roads):
        pass_num = 0
        for road in roads:
            for i in range(3):
                lane = road + '_' + str(i)
                if(lane not in self.now_lane_vehicle.keys()):
                    self.now_lane_vehicle[lane] = set()
                for vehicle in self.pre_lane_vehicle[lane]:
                    if vehicle not in self.now_lane_vehicle[lane] and vehicle in self.vehicles:
                        pass_num += 1
        return pass_num
    
    def _get_reward(self):
        rwds = -self.effective_pressure
        return rwds

    def _update_common_state(self):
        self.lane_vehicle_count = self.eng.get_lane_vehicle_count()
        #vehicle_speed = self.eng.get_vehicle_speed()
        self.lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        self.lane_vehicle = self.eng.get_lane_vehicles()
        # vehicle_speed = self.eng.get_vehicle_speed()
        # vehicle_distance = self.eng.get_vehicle_distance()
        self.effective_count = self.eng.get_lane_effective_vehicle_count(110)
        self.effective_waiting_count = self.eng.get_lane_effective_waiting_vehicle_count(110)
        self.effective_vehicles = self.eng.get_lane_effective_vehicles(110)
    
    def _update_vehicle_intersection(self):
        for lane,vehicles in self.effective_vehicles.items():
            if lane not in self.lane_to_agenti: continue
            for vehicle in vehicles:
                self.vehicle_last_agenti[vehicle] = self.lane_to_agenti[lane]
            
    def _get_observations(self):
        # add 1 dimension to give current step for fixed time agent
        obs = np.full((len(self.agents),self.observation_space.shape[1]),0)
        self.pressure = np.zeros(len(self.agents))
        self.queue = np.zeros(len(self.agents))
        direction = ['turn_left','go_straight','turn_right']
        for ai,agent in enumerate(self.agents):
            roads = self.agents[agent]
            inter = self.intersections[agent]
            for roadlink in inter['roadLinks']:
                inlanei = roadlink['laneLinks'][0]['startLaneIndex']
                inlane = f"{roadlink['startRoad']}_{inlanei}"
                movement = roads.index(roadlink['startRoad'])*3 + direction.index(roadlink['type'])
                if obs[ai][movement] == -1:
                    obs[ai][movement] = 0
                    obs[ai][movement+12] = 0
                
                # effective running 
                obs[ai][movement] += self.effective_count[inlane] - self.effective_waiting_count[inlane]
                # effective pressure
                obs[ai][movement+12] = +self.effective_waiting_count[inlane]
                self.pressure[ai] += self.lane_vehicle_count[inlane]
                self.queue[ai] += self.lane_waiting_vehicle_count[inlane]
                for outlanei in range(3):
                    outlane = f"{roadlink['endRoad']}_{outlanei}"
                    obs[ai][movement+12] -= self.effective_waiting_count[outlane]/3
                    self.pressure[ai]  -= self.lane_vehicle_count[outlane]
                
        self.effective_pressure = obs[:,12:].sum(1)
        onehot_acts = np.zeros((len(self.agents), 8))
        if self.last_action is not None:
            onehot_acts[np.arange(len(self.agents)), self.last_action] = 1  
        obs[:, 24:] = onehot_acts
        

        return obs

    def _get_dones(self):
        dones = np.full((len(self.agents)),self.now_step > self.max_step)
        return dones

    def run_whole_phase(self,action):
        for i,phase in enumerate(action):
            agent_id = self.agentlist[i]
            if action[i] == self.now_phases[i] or self.now_phases[i] == -1:
                self.eng.set_tl_phase(agent_id,phase)
                
            else:
                self.eng.set_tl_phase(agent_id,0)
        #决策间隔小于红灯时间
        if self.seconds_per_step <=self.redtime:
            for t in range(self.seconds_per_step):
                self.eng.next_step()
        #决策间隔大于红灯时间
        else:
            for t in range(self.redtime):
                self.eng.next_step()
            for i,phase in enumerate(action):
                agent_id = self.agentlist[i]
                self.eng.set_tl_phase(agent_id,phase)

            for t in range(self.redtime,self.seconds_per_step):
                self.eng.next_step()
                if t == self.seconds_per_step:
                    if self.reward_flag:
                        self.reward += self._get_reward()
                    
        self.now_step+=self.seconds_per_step
        self.now_phases = action
    def step(self, action):
        # here action is a dict {agent_id:phase}
        self._update_common_state()
        self.last_action = action
        self.reward = np.zeros(len(self.agentlist))
        self.run_whole_phase(action)
        dones = self._get_dones()
        obs = self._get_observations()
        reward = self._get_reward()
        info = {}#self._get_reward()self._get_info()
        self._update_vehicle_intersection()
        if self.now_step == 3600:
            print('average time:{}'.format(self.eng.get_average_travel_time()))
        return obs, reward, dones , info

    def reset(self):
        self.eng = cityflow.Engine(self.eng_config_file, thread_num = 1)
        for t in range(random.randint(0,20)):
            self.eng.next_step()
        #self.eng.reset()
        self.vehicle_last_agenti = {}
        self.now_step = 0
        self.now_phases = np.full((len(self.agents)),-1)
        self._update_common_state()
        obs = self._get_observations()
        return obs

from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
def make_parallel_env(eng_config, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = MyEnv(eng_config)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

