import gym
from gym.logger import info
import numpy as np
import torch
from env.myenv import MyEnv
class MaEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.agents = env.agents
        self.agent_road = list(self.env.agents.values())
        self.speedlimit = {o:self.env.roads[o]['speed_limit'] for o in self.env.roads}
        self.seconds_per_step = self.env.seconds_per_step
        self.att = 0
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.flow_buffer = env.flow_buffer

    def process_obs(self, obs):

        return np.array(obs)

    def step(self, action_n):

        t = self.env.now_step

        action = np.argmax(action_n,1)#

        #obs
        obs, rwd, dones, infos = self.env.step(action)
        obs_n = self.process_obs(obs)

        #print(extra_delay_index)
        dones_n = dones

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

def make_env(config_file,buffer):
    env = MyEnv(config_file,buffer=buffer)
    return MaEnv(env)


from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
def make_parallel_env(config_file, n_rollout_threads, seed, buffer=None):
    def get_env_fn(rank):
        def init_env():
            #env = make_env(env_id, discrete_action=True)
            env = make_env(config_file,buffer)
            
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
