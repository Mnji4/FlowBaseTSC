import gym
from gym.logger import info
import numpy as np
import torch
from env.myenv import MyEnv
data = np.load('data4.npy', allow_pickle=True).item()
cloest = data['cloest']
import random
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
            self.past_action.append([random.randint(1,4) for _ in range(20)])
        self.neighbor_inter = [cloest[o][1:] for o in cloest]
        self.att = 0
        



    def process_obs(self, obs):

        obs_n = []
        for i in range(self.nagents):
            obsi = []

            # 前5个动作

            past_action = self.past_action[i]

            #obsi = [i] + obsi[-1:] + neighbor_action + in_lane_speed_diff + list(obs.values())[i*2+1][1:13]
            def onehot_a(l):
                r = []
                for i in l:
                    o = [0] * 4
                    if i > 0:
                        o[i-1] = 1
                    r = r + o
                return r
            def onehot_o(i):
                o = [0] * self.nagents
                o[i-1] = 1
                return o
            obsi = onehot_a(past_action[-1:])+onehot_o((past_action[-1]%4)+1)+list(obs[i][:24])+list(obs[i][36:44])
            #obsi = [i] + onehot(past_action[-1:] + neighbor_action) + list(obs[i])
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

        action = {}#
        for i in range(self.nagents):
            if action_n[i][0] == 1:
                action[self.agentlist[i]] = (self.past_action[i][-1]%4)+1
                self.past_action[i] = self.past_action[i][1:] + [(self.past_action[i][-1]%4)+1]
            else:
                action[self.agentlist[i]] = self.past_action[i][-1]
            

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
        self.att += rwd.sum()
        if self.env.now_step == 3600:
            print(self.att)
        dones_n = np.array([o for o in list(dones.values())])
        pressure = obs_n[:,-8:-4].sum(1) - obs_n[:,-32:-20].sum(1)
        queue = -obs_n[:,-20:-8].sum(1)
        num_pass = infos
        delay = -obs[:,24:36].sum(1)/10

        reward_n = pressure
        infos = np.array((queue,pressure,delay)).T#,delay,num_pass
        return obs_n, reward_n, dones_n, infos

    def reset(self):
        # reset world
        obs, infos = self.env.reset()
        self.att = 0
        # obs, reward, dones, infos = self.env.step({})
        # print(sum(self.process_rwd(reward)))
        obs_n = self.process_obs(obs)
        
        return obs_n

    def gen_cloest_agents(self):
        pass

def make_env():
    env = MyEnv('manhattan/config_4.json')
    return MaEnv(env)


from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            #env = make_env(env_id, discrete_action=True)
            env = make_env()
            
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