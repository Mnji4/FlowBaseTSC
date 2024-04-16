import gym
import numpy as np
import sys
sys.path.append('CityFlow/dist/CityFlow-0.1.1-py3.7-linux-x86_64.egg')
import cityflow
import json
action_to_onehot = np.array([
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0]
])

class MyEnv(gym.Env):
    def __init__(self, eng_config_file, max_step = 3600, buffer = None):
        f = open(eng_config_file, 'r')
        roadnet_file = json.load(f)['roadnetFile']
        f = open(roadnet_file, 'r')
        a = json.load(f)
        f.close()
        self.eng_config_file = eng_config_file
        self.eng = cityflow.Engine(self.eng_config_file, thread_num=1)
        self.eng.set_save_replay(True)
        self.reward_flag = 1
        self.now_step = 0
        self.max_step = max_step
        self.seconds_per_step = 15
        self.redtime = 5
        self.info_enable = 0
        self.intersections = {}
        self.agents = {}
        self.single_phase_agents = {} #边缘
        self.observation_features = ['lane_vehicle_num', 'waiting_vehicle_num']
        self.inverse_road = {}
        self.last_action = None
        self.action = None

        self.last_lane_vehicles = {}
        self.last_effective_vehicles = {}
        self.passnum = 0
        self.catchnum = 0
        
        self.lane_pass_vehicles = {}
        self.vehicle_duration = {}
        self.sa_history = {}
        self.flow_buffer = buffer
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

        self.agent_inlane = [[str(road) + '_' + str(i) for i in range(3) for road in agentroads[:4]] for agentroads in self.agents.values()]
        self.agent_outlane = [[str(road) + '_' + str(i) for i in range(3) for road in agentroads[4:]] for agentroads in self.agents.values()]
        self.inlane_to_agenti = {}
        for k,v in self.roads.items():
            if v['end_inter'] not in self.agentlist:
                continue
            for i in range(3):
                self.inlane_to_agenti[f"{k}_{i}"] = self.agentlist.index(v['end_inter'])
        self.outlane_to_agenti = {}
        for k,v in self.roads.items():
            if v['start_inter'] not in self.agentlist:
                continue
            for i in range(3):
                self.outlane_to_agenti[f"{k}_{i}"] = self.agentlist.index(v['start_inter'])
        self.reset()
    
    def _get_roadmask(self):
        return

    def _process_passed(self, agenti, outlane):
        if(outlane not in self.last_lane_vehicles):
            return #无车道车辆状态
        if(outlane not in self.inlane_to_agenti):
            return #下游非agent
        now_vehicles = set(self.lane_vehicles[outlane])
        last_vehicles = set(self.last_lane_vehicles[outlane])
        candidates = now_vehicles - last_vehicles
        # 行程在本条路结束的车不算
        for o in candidates:
            if self.vehicle_end_lane[o]!=outlane[:-2]:
                if(outlane not in self.lane_pass_vehicles):
                    self.lane_pass_vehicles[outlane] = {}
                self.lane_pass_vehicles[outlane][o]=self.now_step-self.seconds_per_step
                self.passnum+=1
                 
    def _process_stuck(self, agenti, inlane):
        if(inlane not in self.last_effective_vehicles):
            return
        now_vehicles = set(self.effective_vehicles[inlane])
        last_vehicles = set(self.last_effective_vehicles[inlane])
        n = len(now_vehicles & last_vehicles)
        s = self.sa_history[self.now_step-self.seconds_per_step][0][agenti]
        a = self.sa_history[self.now_step-self.seconds_per_step][1][agenti]
        r = -self.seconds_per_step*n/10
        next_s = self.obs[agenti]
        _obs = np.expand_dims(s,(0,1))
        _actions = np.expand_dims(a,(0,1))
        _rewards = np.expand_dims(r,(0,1))
        _weights = np.full_like(_rewards,1)
        _next_obs = np.expand_dims(next_s,(0,1))
        _dones = np.full_like(_rewards,False)
        self.flow_buffer.push(_obs, _actions, _rewards, _next_obs, _dones,_weights)
    
    def _process_new(self, agenti, inlane):
        if(inlane not in self.last_effective_vehicles):
            return
        if inlane not in self.lane_pass_vehicles: #没统计进放行的
            return
        #按车辆统计新来
        now_vehicles = set(self.effective_vehicles[inlane])
        last_vehicles = set(self.last_effective_vehicles[inlane])
        candidates = (now_vehicles - last_vehicles)
        pass_agent = self.outlane_to_agenti[inlane]
        catch_agent = self.inlane_to_agenti[inlane]
        vehicle_pass_ts = {}
        for o in candidates:
            if o in self.lane_pass_vehicles[inlane]:
                if(inlane not in self.vehicle_duration):
                    self.vehicle_duration[inlane] = {}
                pass_t = self.lane_pass_vehicles[inlane][o]
                self.vehicle_duration[inlane][o] = (pass_t,self.now_step)
                self.lane_pass_vehicles[inlane].pop(o)
                if pass_t not in vehicle_pass_ts:
                    vehicle_pass_ts[pass_t] = 0
                vehicle_pass_ts[pass_t] += 1
                self.catchnum+=1 
        #push to buffer
        for pass_t,n in vehicle_pass_ts.items():
            s = self.sa_history[pass_t][0][pass_agent]
            a = self.sa_history[pass_t][1][pass_agent]
            r = (pass_t-self.now_step)*n/10
            next_s = self.obs[catch_agent]
            _obs = np.expand_dims(s,(0,1))
            _actions = np.expand_dims(a,(0,1))
            _rewards = np.expand_dims(r,(0,1))
            _weights = np.full_like(_rewards,1)
            _next_obs = np.expand_dims(next_s,(0,1))
            _dones = np.full_like(_rewards,False)
            self.flow_buffer.push(_obs, _actions, _rewards, _next_obs, _dones,_weights)


    def _process_passage(self):
        for agenti,lanes in enumerate(self.agent_inlane):
            for inlane in lanes:
                #stuck
                self._process_stuck(agenti,inlane)
                # new, catch semi trasition
        for agenti,lanes in enumerate(self.agent_outlane):
            for outlane in lanes:
                self._process_passed(agenti,outlane)
                self._process_new(agenti,outlane)
                
        self.last_lane_vehicles = self.lane_vehicles
        self.last_effective_vehicles = self.effective_vehicles
        
    def _get_reward(self):
        rwds = -self.effective_pressure*4
        return rwds

    def _update_common_state(self):
        self.lane_vehicle_count = self.eng.get_lane_vehicle_count()
        #vehicle_speed = self.eng.get_vehicle_speed()
        self.lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        self.lane_vehicles = self.eng.get_lane_vehicles()
        # vehicle_speed = self.eng.get_vehicle_speed()
        self.vehicle_distance = self.eng.get_vehicle_distance()
        self.vehicle_now_lane = self.eng.get_vehicle_now_lane()
        self.vehicle_end_lane = self.eng.get_vehicle_end_lane()
        self.effective_count = self.eng.get_lane_effective_vehicle_count(self.seconds_per_step*11.111*1)
        self.effective_waiting_count = self.eng.get_lane_effective_waiting_vehicle_count(self.seconds_per_step*11.111*1)
        self.effective_vehicles = self.eng.get_lane_effective_vehicles(self.seconds_per_step*11.111*1)
        self.vehicles = self.eng.get_vehicles(True)
            
    def _update_vehicle_intersection(self):
        for lane,vehicles in self.effective_vehicles.items():
            if lane not in self.inlane_to_agenti: continue
            for vehicle in vehicles:
                self.vehicle_last_agenti[vehicle] = self.inlane_to_agenti[lane]
            
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
                self.pressure[ai] += self.lane_waiting_vehicle_count[inlane]
                self.queue[ai] += self.lane_waiting_vehicle_count[inlane]
                for outlanei in range(3):
                    outlane = f"{roadlink['endRoad']}_{outlanei}"
                    # obs[ai][movement+12] -= self.effective_waiting_count[outlane]/3
                    self.pressure[ai]  -= self.lane_waiting_vehicle_count[outlane]
                # for v in self.lane_vehicle[inlane]:
                #     distance = self.roads[roadlink['startRoad']]['length'] - self.vehicle_distance[v]
                #     if distance < 111:
                #         cell = int(distance//11.111)
                #         obs[ai][44+movement*10+cell] += 1
                
        self.effective_pressure = obs[:,12:24].sum(1)
        onehot_acts = np.zeros((len(self.agents), 8))
        if self.action is not None and len(self.action):
            onehot_acts = np.take(action_to_onehot, self.action, axis=0) 
        obs[:, 24:32] = onehot_acts
        # onehot_agenti = np.eye(len(self.agents))
        # obs[:, 32:44] = onehot_agenti
        

        return obs

    def _get_dones(self):
        dones = np.full((len(self.agents)),self.now_step > self.max_step)
        return dones

    def run_whole_phase(self,action):
        for i,phase in enumerate(action):
            agent_id = self.agentlist[i]
            if action[i] == self.now_phases[i] or self.now_phases[i] == -1:
                self.eng.set_tl_phase(agent_id,phase+1)
                
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
                self.eng.set_tl_phase(agent_id,phase+1)

            for t in range(self.redtime,self.seconds_per_step):
                self.eng.next_step()
                if t == self.seconds_per_step:
                    if self.reward_flag:
                        self.reward += self._get_reward()
                # try:
                #     info = self.eng.get_vehicle_info('flow_2536_0')
                #     print(f"{info['distance']} {info['road']} {info['intersection']} {info['route']} ")
                # except:
                #     pass
        self.now_step+=self.seconds_per_step
        self.now_phases = action
    def step(self, action):
        # here action is a dict {agent_id:phase}
        self.sa_history[self.now_step] = (self.obs,action)
        self.action = action
        self.reward = np.zeros(len(self.agentlist))
        self.run_whole_phase(action)
        
        self._update_common_state()
        self.dones = self._get_dones()
        self.obs = self._get_observations()
        self.reward = self._get_reward()
        info = {}#self._get_reward()self._get_info()
        # self._update_vehicle_intersection()
        
        self._process_passage()

        if self.now_step == 3600:
            print(f'average time:{self.eng.get_average_travel_time()}')
            print(f'buffer len:{len(self.flow_buffer)}')
        self.last_action = action
        return self.obs, self.reward, self.dones , info

    def reset(self):
        self.eng.reset()# = cityflow.Engine(self.eng_config_file, thread_num = 1)
        # for t in range(random.randint(0,20)):
        #     self.eng.next_step()
        #self.eng.reset()
        self.vehicle_last_agenti = {}
        self.now_step = 0
        self.now_phases = np.full((len(self.agents)),-1)
        self._update_common_state()
        self.obs = self._get_observations()
        return self.obs

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

