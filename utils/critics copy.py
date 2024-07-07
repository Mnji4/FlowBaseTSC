from math import nan
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from utils.misc import onehot_from_logits, categorical_sample, epsilon_greedy
#cloest = np.load("cloest.npy")

#cloest = np.load("cloest.npy")

class MaacCritic(nn.Module):

    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(MaacCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

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

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
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
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets


class GruAttentionCritic(nn.Module):


    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=False, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(GruAttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.hidden_dim = hidden_dim
        
        sdim, adim = sa_sizes[0]
        idim = sdim + adim
        odim = adim

        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.noline = nn.LeakyReLU()
        self.fc = nn.Linear(hidden_dim, odim)

        self.state_encoder = nn.Sequential()
        if norm_in:
            self.state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                        sdim, affine=False))
        self.state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                        hidden_dim))
        self.state_encoder.add_module('s_enc_nl', nn.LeakyReLU())

        self.critic = nn.Sequential()
        self.critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                    hidden_dim))
        # iterate over agents
        self.critic_encoders = nn.ModuleList()

        self.critic_encoder = nn.Sequential()
        if norm_in:
            self.critic_encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                        affine=False))
        self.critic_encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
        self.critic_encoder.add_module('enc_nl', nn.LeakyReLU())


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

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def init_hidden_state(self, n_agents = 16, threads = 4, hidden_states = None):

        if hidden_states is not None:
            self.h = hidden_states
        else:
            self.h = torch.zeros((n_agents * threads, 128))

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, mask=None, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
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
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        
        #sa_encodings = self.critic_encoder(torch.cat(inps)).view(len(inps), -1, self.hidden_dim)
        #sa_encodings = [i for i in sa_encodings]

        sa_encodings = [self.critic_encoder(inp) for inp in inps]

        # extract state encoding for each agent that we're returning Q for

        s_encoding = self.state_encoder(states[0])
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        head_selectors = [sel_ext(s_encoding) for sel_ext in self.selector_extractors]

        other_all_values = []
        all_attend_logits = []
        all_attend_probs = []
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selector in zip(
                all_head_keys, all_head_values, head_selectors):
            # iterate over agents
            keys = [k for j, k in enumerate(curr_head_keys) if j != 0]
            values = [v for j, v in enumerate(curr_head_values) if j != 0]
            # calculate attention across agents
            attend_logits = torch.matmul(curr_head_selector.view(curr_head_selector.shape[0], 1, -1),
                                            torch.stack(keys).permute(1, 2, 0))

            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])

            if mask is not None:
                scaled_attend_logits = scaled_attend_logits.masked_fill(mask, value=torch.tensor(-1e9).to(mask.device))

            attend_weights = F.softmax(scaled_attend_logits, dim=2)
            
            
            other_values = (torch.stack(values).permute(1, 2, 0) *
                            attend_weights).sum(dim=2)
            other_all_values.append(other_values)
            all_attend_logits.append(attend_logits)
            all_attend_probs.append(attend_weights)
        # calculate Q per agent
        all_rets = []
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                            .mean()) for probs in all_attend_probs]
        agent_rets = []
        
        critic_in = torch.cat((s_encoding, *other_all_values), dim=1)
        h1 = self.critic(critic_in)
        h = self.gru(h1,self.h)
        self.h = h.detach()
        h2 = self.fc(h)
        all_q = self.noline(h2)
        int_acs = actions[0].max(dim=1, keepdim=True)[1]
        q = all_q.gather(1, int_acs)
        if return_q:
            agent_rets.append(q)
        if return_all_q:
            agent_rets.append(all_q)
        if regularize:
            # regularize magnitude of attention logits
            attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                        all_attend_logits)
            regs = (attend_mag_reg,)
            agent_rets.append(regs)
        if return_attend:
            agent_rets.append(np.array(all_attend_probs))

        if len(agent_rets) == 1:
            all_rets.append(agent_rets[0])
        else:
            all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets


class AttentionCritic(nn.Module):


    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.hidden_dim = hidden_dim

        
        sdim, adim = sa_sizes[0]
        idim = sdim + adim
        odim = adim
        self.state_encoder = nn.Sequential()
        if norm_in:
            self.state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                        sdim, affine=False))
        self.state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                        hidden_dim))
        self.state_encoder.add_module('s_enc_nl', nn.LeakyReLU())

        self.critic = nn.Sequential()
        self.critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                    hidden_dim))
        self.critic.add_module('critic_nl', nn.LeakyReLU())
        self.critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
        # iterate over agents
        self.critic_encoders = nn.ModuleList()

        self.critic_encoder = nn.Sequential()
        if norm_in:
            self.critic_encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                        affine=False))
        self.critic_encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
        self.critic_encoder.add_module('enc_nl', nn.LeakyReLU())


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

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, mask=None, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
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
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        
        #sa_encodings = self.critic_encoder(torch.cat(inps)).view(len(inps), -1, self.hidden_dim)
        #sa_encodings = [i for i in sa_encodings]

        sa_encodings = [self.critic_encoder(inp) for inp in inps]

        # extract state encoding for each agent that we're returning Q for

        s_encoding = self.state_encoder(states[0])
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        head_selectors = [sel_ext(s_encoding) for sel_ext in self.selector_extractors]

        other_all_values = []
        all_attend_logits = []
        all_attend_probs = []
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selector in zip(
                all_head_keys, all_head_values, head_selectors):
            # iterate over agents
            keys = [k for j, k in enumerate(curr_head_keys) if j != 0]
            values = [v for j, v in enumerate(curr_head_values) if j != 0]
            # calculate attention across agents
            attend_logits = torch.matmul(curr_head_selector.view(curr_head_selector.shape[0], 1, -1),
                                            torch.stack(keys).permute(1, 2, 0))

            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])

            if mask is not None:
                scaled_attend_logits = scaled_attend_logits.masked_fill(mask, value=torch.tensor(-1e9).to(mask.device))

            attend_weights = F.softmax(scaled_attend_logits, dim=2)
            
            
            other_values = (torch.stack(values).permute(1, 2, 0) *
                            attend_weights).sum(dim=2)
            other_all_values.append(other_values)
            all_attend_logits.append(attend_logits)
            all_attend_probs.append(attend_weights)
        # calculate Q per agent
        all_rets = []
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                            .mean()) for probs in all_attend_probs]
        agent_rets = []
        
        critic_in = torch.cat((s_encoding, *other_all_values), dim=1)
        all_q = self.critic(critic_in)
        int_acs = actions[0].max(dim=1, keepdim=True)[1]
        q = all_q.gather(1, int_acs)
        if return_q:
            agent_rets.append(q)
        if return_all_q:
            agent_rets.append(all_q)
        if regularize:
            # regularize magnitude of attention logits
            attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                        all_attend_logits)
            regs = (attend_mag_reg,)
            agent_rets.append(regs)
        if return_attend:
            agent_rets.append(np.array(all_attend_probs))

        if len(agent_rets) == 1:
            all_rets.append(agent_rets[0])
        else:
            all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

class OriginCritic(nn.Module):

    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(OriginCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

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

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
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
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets


class GruAttentionSingleCritic(nn.Module):


    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=False, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(GruAttentionSingleCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.hidden_dim = hidden_dim
        
        sdim, adim = sa_sizes[0]
        idim = sdim + adim
        odim = adim

        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.noline = nn.LeakyReLU()
        self.fc = nn.Linear(hidden_dim, odim)

        self.state_encoder = nn.Sequential()
        if norm_in:
            self.state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                        sdim, affine=False))
        self.state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                        hidden_dim))
        self.state_encoder.add_module('s_enc_nl', nn.LeakyReLU())

        self.critic = nn.Sequential()
        self.critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                    hidden_dim))
        # iterate over agents
        self.critic_encoders = nn.ModuleList()

        self.critic_encoder = nn.Sequential()
        if norm_in:
            self.critic_encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                        affine=False))
        self.critic_encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
        self.critic_encoder.add_module('enc_nl', nn.LeakyReLU())


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

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def init_hidden_state(self, n_agents = 16, threads = 4, hidden_states = None):

        if hidden_states is not None:
            self.h = hidden_states
        else:
            self.h = torch.zeros((n_agents * threads, 128))


    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=False, return_all_q=True,
                regularize=False, return_attend=False, logger=None, niter=0, mask=None, return_act=False, explore=True):

        if agents is None:
            agents = range(len(self.critic_encoders))
        states = inps
        #actions = [a for s, a in inps]
        #inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        
        #sa_encodings = self.critic_encoder(torch.cat(inps)).view(len(inps), -1, self.hidden_dim)
        #sa_encodings = [i for i in sa_encodings]

        sa_encodings = [self.state_encoder(inp) for inp in states]

        # extract state encoding for each agent that we're returning Q for

        s_encoding = sa_encodings[0]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        head_selectors = [sel_ext(s_encoding) for sel_ext in self.selector_extractors]

        other_all_values = []
        all_attend_logits = []
        all_attend_probs = []
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selector in zip(
                all_head_keys, all_head_values, head_selectors):
            # iterate over agents
            keys = [k for j, k in enumerate(curr_head_keys) if j != 0]
            values = [v for j, v in enumerate(curr_head_values) if j != 0]
            # calculate attention across agents
            attend_logits = torch.matmul(curr_head_selector.view(curr_head_selector.shape[0], 1, -1),
                                            torch.stack(keys).permute(1, 2, 0))

            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])

            if mask is not None:
                scaled_attend_logits = scaled_attend_logits.masked_fill(mask, value=torch.tensor(-1e9).to(mask.device))

            attend_weights = F.softmax(scaled_attend_logits, dim=2)
            
            
            other_values = (torch.stack(values).permute(1, 2, 0) *
                            attend_weights).sum(dim=2)
            other_all_values.append(other_values)
            all_attend_logits.append(attend_logits)
            all_attend_probs.append(attend_weights)
        
        # calculate Q per agent
        all_rets = []
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                            .mean()) for probs in all_attend_probs]
        agent_rets = []
        
        #critic_in = torch.cat((s_encoding, *other_all_values), dim=1)
        critic_in = s_encoding
        all_q = self.critic(critic_in)

        # # action mask
        # env_num = all_q.shape[0] // 859

        # self.mask = np.array([[self.phase[self.no_exist[i]] for _ in range(env_num) ] for i in range(859)])

        # self.mask = torch.tensor(self.mask, dtype=torch.float32, device=all_q.device).view(-1, 8)

        # all_q = all_q.masked_fill(self.mask.bool(), torch.tensor(-np.inf, dtype=torch.float32, device=all_q.device))
        
        probs = F.softmax(all_q, dim=1)

        on_gpu = probs.is_cuda

        if explore:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        if return_act:
            agent_rets.append(act)
        if return_q:
            q = all_q.max(dim=1, keepdim=True)[0]
            agent_rets.append(q)
        if return_all_q:
            agent_rets.append(all_q)
        if regularize:
            # regularize magnitude of attention logits
            attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                        all_attend_logits)
            regs = (attend_mag_reg,)
            agent_rets.append(regs)
        if return_attend:
            agent_rets.append(np.array(all_attend_probs))

        if len(agent_rets) == 1:
            all_rets.append(agent_rets[0])
        else:
            all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
        
        critic_in = torch.cat((s_encoding, *other_all_values), dim=1)
        h1 = self.critic(critic_in)
        h = self.gru(h1,self.h)
        self.h = h.detach()
        h2 = self.fc(h)
        all_q = self.noline(h2)
        int_acs = actions[0].max(dim=1, keepdim=True)[1]
        q = all_q.gather(1, int_acs)
        probs = F.softmax(all_q, dim=1)

        on_gpu = probs.is_cuda

        if explore:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        if return_act:
            agent_rets.append(act)
        if return_q:
            q = all_q.max(dim=1, keepdim=True)[0]
            agent_rets.append(q)
        if return_all_q:
            agent_rets.append(all_q)
        if regularize:
            # regularize magnitude of attention logits
            attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                        all_attend_logits)
            regs = (attend_mag_reg,)
            agent_rets.append(regs)
        if return_attend:
            agent_rets.append(np.array(all_attend_probs))

        if len(agent_rets) == 1:
            all_rets.append(agent_rets[0])
        else:
            all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets


class AttentionSingleCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=256, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionSingleCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.hidden_dim = hidden_dim

        
        sdim, adim = sa_sizes[0]
        idim = sdim + adim
        odim = adim
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
        # self.critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
        #                                             hidden_dim))
        self.critic.add_module('critic_nl', nn.LeakyReLU())
        self.critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
        # iterate over agents
        self.critic_encoders = nn.ModuleList()

        self.critic_encoder = nn.Sequential()
        if norm_in:
            self.critic_encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                        affine=False))
        self.critic_encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
        self.critic_encoder.add_module('enc_nl', nn.LeakyReLU())


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

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

        phase = [
            [0,0,0,1,0,1,1,0],
            [0,1,0,0,0,0,1,1],
            [0,0,0,1,1,0,0,1],
            [0,1,0,0,1,1,0,0],
            [1,1,1,1,1,1,1,1]
        ]

        self.phase = -np.array(phase) + 1

        self.no_exist = np.load('no_exist.npy')

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=False, return_all_q=True,
                regularize=False, return_attend=False, logger=None, niter=0, mask=None, return_act=False, explore=True):
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
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = inps
        #actions = [a for s, a in inps]
        #inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        
        #sa_encodings = self.critic_encoder(torch.cat(inps)).view(len(inps), -1, self.hidden_dim)
        #sa_encodings = [i for i in sa_encodings]

        sa_encodings = [self.state_encoder(inp) for inp in states]

        # extract state encoding for each agent that we're returning Q for

        s_encoding = sa_encodings[0]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        head_selectors = [sel_ext(s_encoding) for sel_ext in self.selector_extractors]

        other_all_values = []
        all_attend_logits = []
        all_attend_probs = []
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selector in zip(
                all_head_keys, all_head_values, head_selectors):
            # iterate over agents
            keys = [k for j, k in enumerate(curr_head_keys) if j != 0]
            values = [v for j, v in enumerate(curr_head_values) if j != 0]
            # calculate attention across agents
            attend_logits = torch.matmul(curr_head_selector.view(curr_head_selector.shape[0], 1, -1),
                                            torch.stack(keys).permute(1, 2, 0))

            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])

            if mask is not None:
                scaled_attend_logits = scaled_attend_logits.masked_fill(mask, value=torch.tensor(-1e9).to(mask.device))

            attend_weights = F.softmax(scaled_attend_logits, dim=2)
            
            
            other_values = (torch.stack(values).permute(1, 2, 0) *
                            attend_weights).sum(dim=2)
            other_all_values.append(other_values)
            all_attend_logits.append(attend_logits)
            all_attend_probs.append(attend_weights)
        
        # calculate Q per agent
        all_rets = []
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                            .mean()) for probs in all_attend_probs]
        agent_rets = []
        
        #critic_in = torch.cat((s_encoding, *other_all_values), dim=1)
        critic_in = s_encoding
        all_q = self.critic(critic_in)

        # # action mask
        # env_num = all_q.shape[0] // 859

        # self.mask = np.array([[self.phase[self.no_exist[i]] for _ in range(env_num) ] for i in range(859)])

        # self.mask = torch.tensor(self.mask, dtype=torch.float32, device=all_q.device).view(-1, 8)

        # all_q = all_q.masked_fill(self.mask.bool(), torch.tensor(-np.inf, dtype=torch.float32, device=all_q.device))
        
        probs = F.softmax(all_q, dim=1)

        on_gpu = probs.is_cuda

        if explore:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        if return_act:
            agent_rets.append(act)
        if return_q:
            q = all_q.max(dim=1, keepdim=True)[0]
            agent_rets.append(q)
        if return_all_q:
            agent_rets.append(all_q)
        if regularize:
            # regularize magnitude of attention logits
            attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                        all_attend_logits)
            regs = (attend_mag_reg,)
            agent_rets.append(regs)
        if return_attend:
            agent_rets.append(np.array(all_attend_probs))

        if len(agent_rets) == 1:
            all_rets.append(agent_rets[0])
        else:
            all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

class SingleCritic(nn.Module):


    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=64, norm_in=True):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(SingleCritic, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)

        self.hidden_dim = hidden_dim

        self.epsilon = 0.9
        sdim, adim = sa_sizes[0]
        idim = sdim + adim
        odim = adim
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
        self.critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
        #self.fc2 = nn.Linear(hidden_dim, odim)

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

        if regularize:
            # regularize magnitude of attention logits
            attend_mag_reg = 1e-3 
            regs = (attend_mag_reg,)
            agent_rets.append(regs)
        # if return_attend:
        #     agent_rets.append(np.array(all_attend_probs))

        if len(agent_rets) == 1:
            all_rets.append(agent_rets[0])
        else:
            all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

class MeanCritic(nn.Module):


    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(MeanCritic, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)

        self.hidden_dim = hidden_dim

        
        sdim, adim = sa_sizes
        idim = sdim + adim
        odim = 1
        self.sa_encoder = nn.Sequential()
        if norm_in:
            self.sa_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                        idim, affine=False))
        self.sa_encoder.add_module('s_enc_fc1', nn.Linear(idim,
                                                        hidden_dim))
        self.sa_encoder.add_module('s_enc_nl', nn.LeakyReLU())

        self.critic = nn.Sequential()
        self.critic.add_module('critic_fc1', nn.Linear(hidden_dim,
                                                    hidden_dim))
        self.critic.add_module('critic_nl', nn.LeakyReLU())
        self.critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))

    def forward(self, inps,  agents=None, return_q=False, logger=None):



        
        sa_encoding = self.sa_encoder(inps)

        q = self.critic(sa_encoding)

        return q



