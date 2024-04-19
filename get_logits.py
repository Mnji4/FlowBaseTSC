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
        filename = Path(config.model_path + 'meta_model_ep41.pt')
        meta_model = Distral.init_from_save(filename, load_critic=True)
        model = [meta_model]
        for i in range(2):
            filename = Path(config.model_path + 'model%i_ep41.pt' %(i))
            model_ = Distral.init_from_save(filename, load_critic=True)
            model.append(model_)


    t = 0
    config.n_rollout_threads =1
    r1_list=[]
    r2_list=[]

    tmp = 30

    for ep_i in range(2):
        # print('testing random')
        config.n_rollout_threads = 1
        env = make_parallel_env(config.env_id, config.n_rollout_threads, random.randint(10))
        obs = env.reset()
        
        rand = 0
        logits = []
        acs = []
        for test_t in range(0,3600,config.interval_length):
            torch_obs = [torch.Tensor(ob) for ob in obs]
            # get actions as torch Variables
            #[thread,agent,act]
            if test_t%tmp==0:
                rand +=1

            if 1:
                actions = [model[ep_i].step([torch_obs[i]], explore=True)[0] for i in range(config.n_rollout_threads)][0]
                logit = [model[ep_i].step([torch_obs[i]], explore=True)[1] for i in range(config.n_rollout_threads)][0]
            else:
                probs0 = [model[0].step([torch_obs[i]], explore=False)[1] for i in range(config.n_rollout_threads)]
                probs1 = [model[1].step([torch_obs[i]], explore=False)[1] for i in range(config.n_rollout_threads)]
                sorted, indices = torch.sort(probs0[0][0], descending=True)

                
                mask1 = (sorted[:,0] - sorted[:,1] < 0.2).unsqueeze(-1).expand_as(probs1[0][0]).unsqueeze(0).to(torch.int64)
                

                mask2 = torch.nn.functional.one_hot(indices[:,1],8) + torch.nn.functional.one_hot(indices[:,2],8)
                p1 = torch.masked_fill(probs1[0][0],mask2==0,0)
                a = torch.stack([probs0[0][0],p1],dim = 0)
                b = torch.gather(a,0,mask1)[0]
                # probs = probs0[0][0] + probs1[0][0]
                act = onehot_from_logits(b)
                actions = [act]

            # rearrange actions to be per environment
            #[thread,agent,act]
            actions = [a.numpy() for a in actions]
            next_obs, rewards, dones, infos = env.step(actions)
            obs = next_obs
            if test_t % 5 == 0:
                logits.extend(logit[0].tolist())
                acs.extend([list(a).index(1) for a in actions[0]])
        np.save(r'logit%i.npy'%(ep_i),np.array(logits))
        np.save(r'act%i.npy'%(ep_i),np.array(acs))
        infos = np.array([infos[i][1] for i in range(config.n_rollout_threads)])

        

    quit()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",default='Distral', help="Name of environment")
    parser.add_argument("--model_name",default='test',
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--dqn", default=1, type=int)
    parser.add_argument("--s_dim", default=44, type=int)
    parser.add_argument("--a_dim", default=8, type=int)
    parser.add_argument("--meta_s_dim", default=52, type=int)
    parser.add_argument("--meta_a_dim", default=2, type=int)
    parser.add_argument("--n_agent", default=12, type=int)
    parser.add_argument("--interval_length", default=30, type=int)
    parser.add_argument("--test_interval", default=10, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--n_episodes", default=200, type=int)
    parser.add_argument("--episode_length", default=3600, type=int)
    parser.add_argument("--steps_per_update", default=10, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=128, type=int,
                        help="Batch size for training")    
    parser.add_argument("--meta_batch_size",
                        default=256, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=100000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.0002, type=float)
    parser.add_argument("--q_lr", default=0.0001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--log_num",default=0, type=int)
    parser.add_argument("--load_model", default=True, type=bool)
    parser.add_argument("--model_path", default='/start/manhattan/models/Distral/test/run169/')
    parser.add_argument("--meta_model_path", default='manhattan/models/Distral/test/run89/meta_model.pt')
    config = parser.parse_args()

    run(config, 0)
