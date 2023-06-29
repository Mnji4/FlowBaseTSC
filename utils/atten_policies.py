import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample

import numpy as np
cloest = np.load("cloest.npy")

class SelfAttention(nn.Module):
    def __init__(self, s_dim, hid_dim, n_heads):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        
        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(s_dim, hid_dim)
        self.w_k = nn.Linear(s_dim, hid_dim)
        self.w_v = nn.Linear(s_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))
    
    def forward(self, query, key, value):

        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))[:,0,:]

        x = self.fc(x)

        return x

class AttentionPolicy(nn.Module):
    """
    Attention policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=256, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0, attend_heads=4):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionPolicy, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = [input_dim, out_dim]
        self.nagents = 859
        self.attend_heads = attend_heads


        self.obs_dim = input_dim
        self.hidden_dim = hidden_dim

        self.attention = SelfAttention(input_dim, hidden_dim ,attend_heads)

        # self.l1 = nn.Linear(hidden_dim, hidden_dim)
        # self.n1 = nn.LeakyReLU()
        self.l1 = nn.Linear(hidden_dim, out_dim)

        self.norm_in = norm_in

        if norm_in:

            self.query_bn = nn.BatchNorm1d(input_dim,affine=False)
            self.key_bn = nn.BatchNorm1d(input_dim, affine=False)
            self.value_bn = nn.BatchNorm1d(input_dim, affine=False)

        '''
        self.query_encoder = nn.Sequential()

        if norm_in:
            self.state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(

        self.query_encoder.add_module('q_enc_fc1', nn.Linear(sdim,
                                                        hidden_dim))
        self.query_encoder.add_module('q_enc_nl', nn.LeakyReLU())
        self.query_encoder.add_module('q_enc_fc2', nn.Linear(hidden_dim, hidden_dim))

        self.key_encoder = nn.Sequential()
        self.key_encoder.add_module('k_fc1', nn.Linear(s_dim,
                                                    hidden_dim))
        self.key_encoder.add_module('k_nl', nn.LeakyReLU())
        self.key_encoder.add_module('k_fc2', nn.Linear(hidden_dim, hidden_dim))

        self.value_encoder = nn.Sequential()

        if norm_in:
            self.critic_encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                        affine=False))

        self.value_encoder.add_module('v_fc1', nn.Linear(sdim, hidden_dim))
        self.value_encoder.add_module('v_nl', nn.LeakyReLU())


        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))
        '''

    def forward(self, obs):
        
        if self.norm_in:
            query_in = []
            key_in = []
            value_in = []
            query_5obs, key_5obs, value_5obs = [], [], []
            for a_i in range(self.nagents):

                obs[a_i] = torch.tensor(obs[a_i]).float()

                try:
                    query_in.append(self.query_bn(obs[a_i]))
                    key_in.append(self.key_bn(obs[a_i]))
                    value_in.append(self.value_bn(obs[a_i]))
                except:
                    import pdb
                    pdb.set_trace()
            

            for a_i in range(self.nagents):
                query_5obs.append(torch.stack([query_in[i] for i in cloest[a_i]]))
                key_5obs.append(torch.stack([key_in[i] for i in cloest[a_i]]))
                value_5obs.append(torch.stack([value_in[i] for i in cloest[a_i]]))
            query = torch.stack(query_5obs).permute(0,2,1,3).reshape((-1, 5, self.obs_dim))
            key = torch.stack(key_5obs).permute(0,2,1,3).reshape((-1, 5, self.obs_dim))
            value = torch.stack(value_5obs).permute(0,2,1,3).reshape((-1, 5, self.obs_dim))
        else:
            curr_5obs = []
            for a_i in range(self.nagents):
                curr_5obs.append([obs[i] for i in cloest[a_i]])
            
            obs = np.array(curr_5obs).transpose(0,2,1,3).reshape((-1, 5, self.obs_dim))
            obs = torch.tensor(obs).float()
            query, key, value = obs, obs, obs


        h1 = self.attention(query,key,value)
        out = self.l1(F.leaky_relu(h1))

        return out


class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """

        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.nonlin(self.fc3(h2))
        return out


class DiscretePolicy(AttentionPolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets
