import gym
from gym.logger import info
import numpy as np
import torch
from env.myenv import MyEnv
class MaEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.agents = env.agents
        self.agentlist = list(self.env.agents.keys())
        self.agent_road = list(self.env.agents.values())
        self.speedlimit = {o:self.env.roads[o]['speed_limit'] for o in self.env.roads}
        self.nagents = len(self.agentlist)
        self.past_action = []
        self.seconds_per_step = self.env.seconds_per_step
        for i in range(len(self.agentlist)):
            self.past_action.append([0 for _ in range(20)])
        self.att = 0

        

    def process_demand(self, info):
        demand = np.zeros((self.nagents, 36))
        for car_info in info.values():
            t = 30
            route_i = 0
            route_len = len(car_info['route'])
            while t>20 and route_i < route_len - 1:
                if route_i == 0:
                    to_drive = self.env.roads[int(car_info['route'][route_i])]['length']-car_info['distance'][0]
                    
                else:
                    to_drive = self.env.roads[int(car_info['route'][route_i])]['length']
                
                t -= to_drive/(self.env.roads[car_info['route'][route_i]]['speed_limit'])
                
                cur_road = car_info['route'][route_i]
                
                if cur_road in self.agent_inroad_list:
                    
                    next_road = car_info['route'][route_i+1]

                    agent_idx = self.outroad_to_agent[next_road]

                    lane_direct = 3 - ((self.agent_road[agent_idx].index(next_road) - 4) - self.agent_road[agent_idx].index(cur_road))%4

                    lane_idx = self.agent_road[agent_idx].index(cur_road)*3 + lane_direct

                    demand[agent_idx][lane_idx] +=  1

                route_i += 1
            
            while t>10 and route_i < route_len - 1:
                if route_i == 0:
                    to_drive = self.env.roads[int(car_info['route'][route_i])]['length']-car_info['distance'][0]
                    
                else:
                    to_drive = self.env.roads[int(car_info['route'][route_i])]['length']
                
                t -= to_drive/(self.env.roads[car_info['route'][route_i]]['speed_limit'])
                
                cur_road = car_info['route'][route_i]

                if cur_road in self.agent_inroad_list:
                    
                    next_road = car_info['route'][route_i+1]

                    agent_idx = self.outroad_to_agent[next_road]

                    lane_direct = 3 - ((self.agent_road[agent_idx].index(next_road) - 4) - self.agent_road[agent_idx].index(cur_road))%4

                    lane_idx = self.agent_road[agent_idx].index(cur_road)*3 + lane_direct

                    demand[agent_idx][lane_idx+12] +=  1

                route_i += 1

            while t>0 and route_i < route_len - 1:
                if route_i == 0:
                    to_drive = self.env.roads[int(car_info['route'][route_i])]['length']-car_info['distance'][0]
                    
                else:
                    to_drive = self.env.roads[int(car_info['route'][route_i])]['length']
                
                t -= to_drive/(self.env.roads[car_info['route'][route_i]]['speed_limit'])
                
                cur_road = car_info['route'][route_i]

                if cur_road in self.agent_inroad_list:
                    
                    next_road = car_info['route'][route_i+1]

                    agent_idx = self.outroad_to_agent[next_road]

                    lane_direct = 3 - ((self.agent_road[agent_idx].index(next_road) - 4) - self.agent_road[agent_idx].index(cur_road))%4

                    lane_idx = self.agent_road[agent_idx].index(cur_road)*3 + lane_direct

                    demand[agent_idx][lane_idx+24] +=  1

                route_i += 1
                    
        return demand 

    def process_delay(self, info):
        
        delay_each_car = np.zeros(126669)

        cur_step_lane_delay = np.zeros((self.nagents, 12)) #lane:delay
        for car_id in info:
            #id = str(int(car_info['route'][-1])) + '_' + str(int(car_info['start_time'][0])) + '_' + str(car_info['t_ff'][0])
            #new car
            car_info = info[car_id]
            if self.cars[car_id] == 0:
                
                cur_road = int(car_info['road'][0])

                try:
                    ind = self.id_route[car_id][0].index(cur_road)
                except:
                    self.id_route[car_id][0] = car_info['route']

                    tmp = 0
                    distance_list = []
                    for road_id in car_info['route']:
                        tmp += self.env.roads[road_id]['length']
                        distance_list.append(tmp)
                    self.id_route[car_id][1] = distance_list

                    ind = self.id_route[car_id][0].index(cur_road)


                if ind == 0:
                    cur_dis = car_info['distance'][0]
                else:
                    cur_dis = self.id_route[car_id][1][ind-1] + car_info['distance'][0]

                trip_length = cur_dis

                idea_time = trip_length / (self.env.roads[cur_road]['speed_limit'])

                delay = 10 - idea_time

                delay_each_car[car_id] = delay

                # process delay_index
            else:
                
                cur_road = int(car_info['road'][0])

                try:
                    ind = self.id_route[car_id][0].index(cur_road)
                except:
                    self.id_route[car_id][0] = car_info['route']

                    tmp = 0
                    distance_list = []
                    for road_id in car_info['route']:
                        tmp += self.env.roads[road_id]['length']
                        distance_list.append(tmp)
                    self.id_route[car_id][1] = distance_list

                    ind = self.id_route[car_id][0].index(cur_road)

                if ind == 0:
                    cur_dis = car_info['distance'][0]
                else:
                    cur_dis = self.id_route[car_id][1][ind-1] + car_info['distance'][0]
                former_dis = self.cars[car_id][0]

                trip_length = cur_dis - former_dis

                


                former_road = int(self.cars[car_id][1]['road'][0])

                if cur_road != former_road:

                    cur_road_ind = self.id_route[car_id][0].index(cur_road)
                    
                    init_road_ind = self.id_route[car_id][0].index(int(self.cars[car_id][1]['road'][0]))
                    
                    idea_time = 0

                    for road_ind in range(init_road_ind, cur_road_ind+1):
                        
                        road_id = self.id_route[car_id][0][road_ind]

                        if road_ind == init_road_ind:

                            dis = self.id_route[car_id][1][road_ind] - self.cars[car_id][0]

                            idea_time += dis / self.env.roads[road_id]['speed_limit']

                        elif road_ind == cur_road_ind:

                            dis = car_info['distance'][0]

                            idea_time += dis / self.env.roads[cur_road]['speed_limit']

                        else:

                            idea_time += self.env.roads[road_id]['length'] / self.env.roads[road_id]['speed_limit']
                    
                    delay = 10 - idea_time

                    delay_each_car[car_id] = delay

                else:

                    idea_time = trip_length / (self.env.roads[cur_road]['speed_limit'])

                    delay = 10 - idea_time

                    delay_each_car[car_id] = delay

            self.cars[car_id] = [cur_dis, car_info]
        
        # process served cars
        for car_id in range(126669):
            if self.cars[car_id] == 0 or self.cars[car_id] == -1:
                continue
            # car served
            if car_id not in info:
                
                trip_length = self.id_route[car_id][1][-1] - self.cars[car_id][0]

                idea_time += trip_length / self.env.roads[int(self.cars[car_id][1]['road'][0])]['speed_limit']

                delay = 10 - idea_time

                delay_each_car[car_id] = delay
                
                self.cars[car_id] = -1



        other_delay = 0

        for car_id in range(126669):
            if self.cars[car_id] == 0 or self.cars[car_id] == -1:
                continue
            if int(self.cars[car_id][1]['road'][0]) in self.agent_inroad_list:

                cur_lane = int(self.cars[car_id][1]['drivable'][0])

                agent_id, lane_index = self.inlane_to_agent[cur_lane]

                cur_step_lane_delay[self.agent_to_index[agent_id],lane_index-1] += delay_each_car[car_id]
        
            else:

                other_delay += delay_each_car[car_id]

        reward = cur_step_lane_delay.sum(1).reshape(859) + np.full((859),other_delay/859)
        reward = -reward



        if self.env.now_step == 3600:
            l = np.array(self.id_route, dtype=object)
            np.save('id_route.npy',l)



        return reward

    def process_obs_queue(self, info):

        # min_road_length = 30, max_road_length = 4313
        # min_agent_road_length = 30 , max_agent_road_length = 3322

        queue = np.zeros((self.nagents, 24))
        for vehicle_info in info.values():
            speed = vehicle_info['speed'][0]
            if speed < 1:
                if int(vehicle_info['road'][0]) in self.agent_inroad_list:
                    road = vehicle_info['road'][0]
                    distance = vehicle_info['distance'][0]
                    cur_lane = int(vehicle_info['drivable'][0])
                    agent_id, lane_index = self.inlane_to_agent[cur_lane]
                    queue[self.agent_to_index[agent_id],lane_index-1] += 1
                if int(vehicle_info['road'][0]) in self.agent_outroad_list:
                    road = vehicle_info['road'][0]
                    distance = vehicle_info['distance'][0]
                    cur_lane = int(vehicle_info['drivable'][0])
                    agent_id, lane_index = self.outlane_to_agnet[cur_lane]
                    queue[self.agent_to_index[agent_id],lane_index-1] += 1


        return queue

    def process_obs(self, obs):

        # obs_n = []
        # for i in range(self.nagents):
        #     obsi = []

        #     # 前5个动作

        #     past_action = self.past_action[i]
        #     def onehot_a(l):
        #         r = []
        #         for i in l:
        #             o = [0] * 8
        #             if i > 0:
        #                 o[i-1] = 1
        #             r = r + o
        #         return r
        #     def onehot_o(i):
        #         o = [0] * self.nagents
        #         o[i] = 1
        #         return o
        #     obsi = onehot_a(past_action[-1:])+list(obs[i][:24])
        #     #obsi = onehot_a(past_action[-1:])+list(obs[i][:24])+list(obs[i][36:48]/1000)+list(obs[i][48:56])#+list(obs[i][60:]/10)
            
        #     #obsi = onehot_a(past_action[-1:])+list(obs[i][:24])+list(obs[i][48:56])
        #     #obsi = [i] + onehot(past_action[-1:] + neighbor_action) + list(obs[i])
        #     obs_n.append(obsi)
        # obs_n = np.array(obs_n)
        return np.array(obs)


    def process_rwd(self, rwd):
        rew = list(rwd.values())
        
        rew_n = []
        for ra in rew:
            in_roads = np.array([np.array(o) for o in ra[:12] if o != -1])
            in_num = in_roads[:,1].sum()
            # out_roads = np.array([np.array(o) for o in ra[12:] if o != -1])
            # out_num = out_roads[:,0].sum()
            rew_n.append(in_num)
        return rew_n

    def step(self, action_n):

        t = self.env.now_step
        #step
        #action = {self.agentlist[i]:(np.argmax(action_n[i]*np.array(mask[no_exist[i]]))+1) for i in range(self.nagents)}

        action = np.argmax(action_n,1)#
        for i in range(self.nagents):
            self.past_action[i] = self.past_action[i][1:] + [np.argmax(action_n[i])+1]

        #obs
        obs, rwd, dones, infos = self.env.step(action)
        obs_n = self.process_obs(obs)
        #demand = self.process_demand(infos)
        #obs_n = np.concatenate((obs_n,demand),axis=1)

        #reward_n = self.process_rwd(rwd)
        

                #avg car velocity of each intersection
        #reward_n = []
        # for i in range(self.nagents):
        #     r = np.dot(obs_n[i][6:18],obs_n[i][18:30])
        #     reward_n.append(r)

        
        # for i in range(self.nagents):

        #     inter_delay = 0
        #     for roadi in range(4):
        #         road_delay = 0
        #         if self.agents[i][roadi] != -1:
        #             road_speed_limit = 11.11
        #             for lanei in range(3):
        #                 road_delay += (road_speed_limit-obs_n[i][roadi*3+lanei+51]) * obs_n[i][roadi*3+lanei+]
        #             inter_delay += road_delay
        #     r = -inter_delay
        #     reward_n.append(r)
        #reward_n = self.process_delay(infos)

        #reward_n = obs_n[:,-8:-4].sum(1) - obs_n[:,-32:-20].sum(1)
        #reward_n = -obs_n[:,-20:-8].sum(1)

        #reward_n = -rwd


        extra_delay_index = -obs[:,36:48].sum(1)/10
        #print(extra_delay_index)
        dones_n = dones
        pressure = obs[:,48:52].sum(1) - obs[:,:12].sum(1)
        queue = -obs[:,12:24].sum(1)
        #num_pass = infos-200
        delay = -obs[:,24:36].sum(1)/100

        reward_n = rwd
        infos_n = []# 
        #infos_n = [(queue, 0),()]# 
        #infos = np.array((queue,extra_delay_index)).T

        return obs_n, reward_n, dones_n, infos_n

    def reset(self):
        # reset world
        obs = self.env.reset()
        # obs, reward, dones, infos = self.env.step({})
        # print(sum(self.process_rwd(reward)))
        obs_n = self.process_obs(obs)
        
        return obs_n

    def gen_cloest_agents(self):
        pass

def make_env(config_file):
    env = MyEnv(config_file)
    return MaEnv(env)


from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
def make_parallel_env(config_file, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            #env = make_env(env_id, discrete_action=True)
            env = make_env(config_file)
            
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

#ori_rew

        # for l in list(rwd.values()):
        #     reward_n.append(np.sum([o for o in l if o != -1]))

        #avg car velocity of each intersection

        # for i in range(self.nagents):
        #     vsum = 0
        #     nsum = 0
        #     inter_n = 0
        #     ratiosum = 0
        #     for roadi in range(8):
                
        #         if self.agent_road[i][roadi] != -1:
        #             maxv = 22.23#self.speedlimit[self.agent_road[i][roadi]]
        #             lanevsum = 0
        #             roadn = 1e-5
        #             for lanei in range(3):
        #                 ncar = obs_n[i][roadi*3+lanei+24]
        #                 if ncar > 0:
        #                     lanevsum += ncar * obs_n[i][roadi*3+lanei]
        #                     roadn += ncar
        #             inter_n += roadn
        #             roadv = lanevsum/roadn
        #             roadratio = roadv/maxv
        #             ratiosum += roadratio
        #     if inter_n > 1:
        #         r = ratiosum/inter_n -1
        #     else:
        #         r = 0

        #     reward_n.append(r)