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
        for i in range(len(self.agentlist)):
            self.past_action.append([0 for _ in range(5)])
        #self.neighbor_inter = np.load('ori_neighbor.npy')
        #self.id_route = [[[],[]] for _ in range(126669)]
        
        # self.inlane_to_agent = {}
        # self.outlane_to_agent = {}        
        # self.inroad_to_agent = {}
        # self.outroad_to_agent = {}
        # self.agent_to_index = {}
        # for index, agent_id in enumerate(self.agentlist):
        #     self.agent_to_index[agent_id] = index

        
        # # 计算进出口道的车辆
        # for agent_id, road_list in self.env.agents.items():
        #     for road_index, road in enumerate(road_list[:4]):
        #         if road != -1:
        #             self.inroad_to_agent[int(road)] = self.agent_to_index[agent_id]
        #             for lane_index, lane_id in enumerate(env.roads[road]['lanes']):
        #                 self.inlane_to_agent[int(lane_id)] = [agent_id, road_index * 3 + lane_index + 1]

        #     for road_index, road in enumerate(road_list[4:8]):
        #         if road != -1:
        #             self.outroad_to_agent[int(road)] = self.agent_to_index[agent_id]
        #             for lane_index, lane_id in enumerate(env.roads[road]['lanes']):
        #                 self.outlane_to_agent[int(lane_id)] = [agent_id, road_index * 3 + lane_index + 13]

        # self.agent_inroad_list = []
        # for agent_id, road_list in self.agents.items():
        #     self.agent_inroad_list.extend([int(i) for i in road_list[:4] if i != -1])
        # self.agent_outroad_list = []
        # for agent_id, road_list in self.agents.items():
        #     self.agent_outroad_list.extend([int(i) for i in road_list[4:8] if i != -1])

        # self.cars = [0 for _ in range(126669)]
        #self.id_route = np.load('id_route.npy',allow_pickle=True).tolist()
        # self.neighbor_inter = {}
        # for i in range(len(self.agentlist)):
        #     neighbor = []
        #     for road in self.agent_road[i][:4]:
        #         if road==-1:
        #             neighbor.append(-1)
        #         else:
        #             start_inter = self.env.roads[road]['start_inter']
        #             if start_inter in self.agentlist:
        #                 neighbor.append(start_inter)
        #             else:
        #                 neighbor.append(0)
        #     self.neighbor_inter[self.agentlist[i]] = neighbor
        # self.neighbor_inter = []
        # for i in range(len(self.agentlist)):
        #     neighbor = []
        #     for road in self.agent_road[i][:4]:
        #         if road==-1:
        #             neighbor.append(-1)
        #         else:
        #             start_inter = self.env.roads[road]['start_inter']
        #             if start_inter in self.agentlist:
        #                 neighbor.append(start_inter)
        #             else:
        #                 neighbor.append(0)
        #     self.neighbor_inter.append(neighbor)
        # np.save("neighbor_22.npy",self.neighbor_inter)
        

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

        obs_n = []
        for i in range(self.nagents):
            # neighbor_action = []
            # for inter in self.neighbor_inter[i]:
            #     if inter <= 0:
            #         neighbor_action.append(inter)
            #     else:
            #         neighbor_action.append(self.past_action[self.agentlist.index(inter)][-1])
            # obsi = []

            # 前5个动作

            obsi = self.past_action[i]
            #obsi = [i] + obsi[-1:] + neighbor_action + list(obs.values())[i*2+1][1:]
            # in_lane_speed = list(obs.values())[i*2][1:13]
            # in_lane_speed_diff = [0 for _ in range(12)]
            # for roadi in range(4):
            #     if self.agent_road[i][roadi] != -1:
            #         road_speed_limit = self.speedlimit[self.agent_road[i][roadi]]
            #         for lanei in range(3):
            #             in_lane_speed_diff[roadi*3+lanei] = (in_lane_speed[roadi*3+lanei]-road_speed_limit)

            #obsi = [i] + obsi[-1:] + neighbor_action + in_lane_speed_diff + list(obs.values())[i*2+1][1:13]
            onehot = [0,0,0,0,0,0,0,0]
            onehot[obsi[-1]-1] = 1
            obsi = [i] + onehot + list(obs[i])
            # obsi = neighbor_action + obsi
            obs_n.append(obsi)
        obs_n = np.array(obs_n)
        return np.array(obs_n)


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

        action = {self.agentlist[i]:(np.argmax(action_n[i])+1) for i in range(self.nagents)}

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

        # reward_n = []
        # for i in range(self.nagents):

        #     inter_delay = 0
        #     for roadi in range(4):
        #         road_delay = 0
        #         if self.agent_road[i][roadi] != -1:
        #             road_speed_limit = self.speedlimit[self.agent_road[i][roadi]]
        #             for lanei in range(3):
        #                 road_delay += (road_speed_limit-obs_n[i][roadi*3+lanei+6]) * obs_n[i][roadi*3+lanei+]
        #             inter_delay += road_delay
        #     r = -inter_delay
        #     reward_n.append(r)
        #reward_n = self.process_delay(infos)
        #reward_n = list(queue[:,12:].sum(1) - queue[:,:12].sum(1))
        #reward_n = list(obs_n[:,42:54].sum(1) - obs_n[:,30:42].sum(1))
        #reward_n = list(obs_n[:,14:26].sum(1) - obs_n[:,2:14].sum(1))
        #reward_n = obs_n[:,-8:-4].sum(1) - obs_n[:,2:14].sum(1)
        reward_n =- obs_n[:,-20:-8].sum(1)

        dones_n = np.array([o for o in list(dones.values())])
        return obs_n, reward_n, dones_n, infos

    def reset(self):
        # reset world
        obs, infos = self.env.reset()
        # obs, reward, dones, infos = self.env.step({})
        # print(sum(self.process_rwd(reward)))
        obs_n = self.process_obs(obs)
        return obs_n

    def gen_cloest_agents(self):
        pass

def make_env():
    env = MyEnv('manhattan/config_3.json')
    return MaEnv(env)


from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
def make_parallel_env(eng_config, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            #env = make_env(env_id, discrete_action=True)
            env = MyEnv(eng_config)
            
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