import argparse
from numpy import random
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBufferTime
# from algorithms.attention_sac1 import AttentionSAC
#from algorithms.attention_ppo1 import AttentionPPO
from algorithms.distral import Distral
from algorithms.conditional import Conditional
import gym
import json
from utils.ma_env_time import MaEnv,make_env,make_parallel_env
from utils.misc import onehot_from_logits
#gym


    

def run(config, start = 0):


    torch.manual_seed(1)
    np.random.seed(random.randint(100))
    
    model = []
    if config.load_model:
        for i in range(2):
            filename = Path(config.model_path)
            model_ = Distral.init_from_save(filename, load_critic=True)
            model_.prep_rollouts(device='cuda')
            model.append(model_)
            
    config.n_rollout_threads =1

    for ep_i in range(1):
        # print('testing random')
        config.n_rollout_threads = 1
        env = make_parallel_env(config.env_config, config.n_rollout_threads, random.randint(10))
        obs = env.reset()
        
        for test_t in range(0,3600,env.seconds_per_step):
            torch_obs = [torch.Tensor(ob).cuda() for ob in obs]
            # get actions as torch Variables
            #[thread,agent,act]
            actions = [model[i].step(torch_obs[i], explore=False)[0] for i in range(config.n_rollout_threads)]

            # rearrange actions to be per environment
            #[thread,agent,act]
            actions = [a.cpu().numpy() for a in actions]
            next_obs, rewards, dones, infos = env.step(actions)
            obs = next_obs





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", default=True, type=bool)
    parser.add_argument("--env_config", default='config/test/config_jinan_test.json')
    parser.add_argument("--model_path", default="models/Intersec/run1/model_ep21.pt")
    config = parser.parse_args()

    run(config, 0)
