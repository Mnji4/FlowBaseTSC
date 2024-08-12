from math import nan
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from utils.misc import onehot_from_logits, categorical_sample, epsilon_greedy

class SingleCritic(nn.Module):


    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sdim, adim, hidden_dim=64, norm_in=True):
        """
        Inputs:
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.epsilon = 0.9
        out_dim = adim
        self.state_encoder = nn.Sequential()
        if norm_in:
            self.state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                        sdim, affine=False))
        self.state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                        hidden_dim))
        self.state_encoder.add_module('s_enc_nl', nn.LeakyReLU())

        self.critic = nn.Sequential()
        self.critic.add_module('critic_fc1', nn.Linear(hidden_dim,
                                                    hidden_dim))
        self.critic.add_module('critic_nl', nn.LeakyReLU())
        self.critic.add_module('critic_fc2', nn.Linear(hidden_dim, out_dim))
        #self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, inps, mask=None, agents=None, return_q=False, return_all_q=False,return_probs = False,return_logits=False,
                regularize=False, return_attend=False,return_act=False, explore=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        states = inps

        s_encoding = self.state_encoder(states)
        all_rets = []
        agent_rets = []
        
        #all_q = self.critic(s_encoding)
        
        h = self.critic[0](s_encoding)
        logits = self.critic[1](h)
        all_q = self.critic[2](logits)
        
        probs = F.softmax(all_q, dim=1)
        if torch.any(torch.isnan(probs)):
            print()
        on_gpu = probs.is_cuda

        if explore:
            
            int_acs, act = categorical_sample(probs, use_cuda=on_gpu) # epsilon_greedy(self.epsilon, probs, use_cuda=on_gpu) 
        else:
            act = onehot_from_logits(probs)

        if return_act:
            agent_rets.append(act)

        if return_probs:
            agent_rets.append(probs)
        if return_q:
            q = all_q.max(dim=1, keepdim=True)[0]
            agent_rets.append(q)
        if return_all_q:
            agent_rets.append(all_q)
        if return_logits:
            agent_rets.append(logits)

        if len(agent_rets) == 1:
            all_rets.append(agent_rets[0])
        else:
            all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets 

class PairCritic(nn.Module):
    def __init__(self, sdim, adim, hidden_dim=64, norm_in=True):
        super().__init__()
    
        self.hidden_dim = hidden_dim

        self.epsilon = 0.9
        out_dim = adim**2
        self.state_encoder1 = nn.Sequential()
        if norm_in:
            self.state_encoder1.add_module('s_enc1_bn', nn.BatchNorm1d(
                                        sdim, affine=False))
        self.state_encoder1.add_module('s_enc1_fc1', nn.Linear(sdim,
                                                        hidden_dim))
        self.state_encoder1.add_module('s_enc1_nl', nn.LeakyReLU())

        self.state_encoder2 = nn.Sequential()
        if norm_in:
            self.state_encoder2.add_module('s_enc2_bn', nn.BatchNorm1d(
                                        sdim, affine=False))
        self.state_encoder2.add_module('s_enc2_fc1', nn.Linear(sdim,
                                                        hidden_dim))
        self.state_encoder2.add_module('s_enc2_nl', nn.LeakyReLU())

        self.critic = nn.Sequential()
        self.critic.add_module('critic_fc1', nn.Linear(hidden_dim,
                                                    hidden_dim))
        self.critic.add_module('critic_nl', nn.LeakyReLU())
        self.critic.add_module('critic_fc2', nn.Linear(hidden_dim, out_dim))

    def forward(self, inps, return_q=False, return_all_q=False,return_probs = False,return_logits=False,
                regularize=False, return_act=False, explore=False, ):

        states1,states2 = inps

        s_encoding1 = self.state_encoder1(states1)
        s_encoding2 = self.state_encoder1(states2)
        all_rets = []
        agent_rets = []
        
        #all_q = self.critic(s_encoding)
        
        h = self.critic[0](s_encoding1+s_encoding2)
        logits = self.critic[1](h)
        all_q = self.critic[2](logits)
        
        probs = F.softmax(all_q, dim=1)
        if torch.any(torch.isnan(probs)):
            print()
        on_gpu = probs.is_cuda

        if explore:
            
            int_acs, act = categorical_sample(probs, use_cuda=on_gpu) # epsilon_greedy(self.epsilon, probs, use_cuda=on_gpu) 
        else:
            act = onehot_from_logits(probs)

        if return_act:
            agent_rets.append(act)

        if return_probs:
            agent_rets.append(probs)
        if return_q:
            q = all_q.max(dim=1, keepdim=True)[0]
            agent_rets.append(q)
        if return_all_q:
            agent_rets.append(all_q)
        if return_logits:
            agent_rets.append(logits)

        if len(agent_rets) == 1:
            all_rets.append(agent_rets[0])
        else:
            all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets