import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic
import numpy as np
MSELoss = torch.nn.MSELoss()

from copy import deepcopy


class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4, dueling=False, norm_in=False,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)
        self.s_dim = sa_size[0][0]
        self.a_dim = sa_size[0][1]
        self.central_agent = AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **agent_init_params[0])
                                    
        # self.agents = [AttentionAgent(lr=pi_lr,
        #                               hidden_dim=pol_hidden_dim,
        #                               **params)
        #                  for params in agent_init_params]
        self.critic = AttentionCritic(sa_size[:5], hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size[:5], hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        #self.init_from_central_agent()
        # self.grad = AttentionAgent(lr=pi_lr,
        #                               hidden_dim=pol_hidden_dim,
        #                               **agent_init_params[0])
    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        # return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
        #                                                        observations)]

        #before = [a.step(obs) for a, obs in zip(self.agents, observations)]


        obs = torch.cat(observations)
        act =self.central_agent.step(obs, explore=explore)
        act = act.view(self.nagents,-1,self.a_dim)

        return [act[i] for i in range(self.nagents)]


    def update_critic(self, sample,cloest, mask, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """

        obs, acs, rews, next_obs, dones = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        pi = self.central_agent.target_policy

        agent_num = len(next_obs)
        batch_size = next_obs[0].shape[0]

        with torch.no_grad():
            tmp = torch.stack(next_obs).view(-1, self.s_dim)
            next_acs, next_log_pis = pi(tmp, return_log_pi=True)
            next_acs = next_acs.view(agent_num, batch_size, -1)
            #next_log_pis = next_log_pis.view(agent_num, batch_size, -1)

        next_5obs = []
        next_5acs = []
        tmp_obs = torch.tensor(np.zeros(next_obs[0].shape), dtype=torch.float32, device=next_obs[0].device)
        tmp_acs = torch.tensor(np.zeros(next_acs[0].shape), dtype=torch.float32, device=next_obs[0].device)
        next_mask_obs = []
        # for a_i in range(self.nagents):
        #     next_5obs.append(torch.stack([next_obs[i] if i == cloest[0] else tmp_obs for i in cloest[a_i]]))
        #     next_5acs.append(torch.stack([next_acs[i] if i == cloest[0] else tmp_acs for i in cloest[a_i]]))
        for a_i in range(self.nagents):
            next_5obs.append(torch.stack([next_obs[i] if i != -1 else tmp_obs for i in cloest[a_i]]))
            next_5acs.append(torch.stack([next_acs[i] if i != -1 else tmp_acs for i in cloest[a_i]]))
            next_mask_obs.append([mask[a_i] for _  in range(batch_size)])
        
        next_mask_obs = torch.ByteTensor(next_mask_obs).bool().to(next_obs[0].device).view(-1, 1, len(mask[0]))
        
        #5 * (threads*agents) *54
        trgt_critic_in = list(zip(torch.cat(next_5obs,dim = 1), torch.cat(next_5acs,dim = 1)))

        corrent_5obs = []
        corrent_5acs = []
        mask_obs = []
        # for a_i in range(self.nagents):
        #     corrent_5obs.append(torch.stack([obs[i] if i == cloest[0] else tmp_obs for i in cloest[a_i]]))
        #     corrent_5acs.append(torch.stack([acs[i] if i == cloest[0] else tmp_acs for i in cloest[a_i]]))
        for a_i in range(self.nagents):
            corrent_5obs.append(torch.stack([obs[i] if i != -1 else tmp_obs for i in cloest[a_i]]))
            corrent_5acs.append(torch.stack([acs[i] if i != -1 else tmp_acs for i in cloest[a_i]]))
            mask_obs.append([mask[a_i] for _  in range(batch_size)])
        
        mask_obs = torch.ByteTensor(mask_obs).bool().to(next_obs[0].device).view(-1,1,len(mask[0]))
        
        critic_in = list(zip(torch.cat(corrent_5obs,dim = 1), torch.cat(corrent_5acs,dim = 1)))
        #trgt_critic_in = list(zip(next_obs, next_acs))
        #critic_in = list(zip(obs, acs))

        next_qs = self.target_critic(trgt_critic_in, mask = next_mask_obs)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter, mask=mask_obs)
        q_loss = 0
        
        pq, regs = critic_rets
        
        target_q = (torch.cat(rews).view(-1, 1) + self.gamma * next_qs *  (1 - torch.cat(dones).view(-1, 1)))
        if soft:
            target_q -= next_log_pis / self.reward_scale
        q_loss += MSELoss(pq, target_q.detach())
        
        ''' if regularization
        for reg in regs:
            q_loss += reg  # regularizing attention
        '''

        q_loss.backward()

        # original maac 10 * self.nagents
        # grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.nagents)
        
        
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            #logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample,cloest,mask,soft=True, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample
        # samp_acs = []
        # all_probs = []
        # all_log_pis = []
        # all_pol_regs = []

        # for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
        #     curr_ac, probs, log_pi, pol_regs, ent = pi(
        #         ob, return_all_probs=True, return_log_pi=True,
        #         regularize=True, return_entropy=True)
        #     logger.add_scalar('policy_entropy' %ent,
        #                       self.niter)
        #     samp_acs.append(curr_ac)
        #     all_probs.append(probs)
        #     all_log_pis.append(log_pi)
        #     all_pol_regs.append(pol_regs)
        obs1 = torch.cat(obs)
        samp_acs,probs,log_pis,pol_regs =  self.central_agent.policy(
                obs1, return_all_probs=True, return_log_pi=True,
                regularize=True)

        # critic_in = []
        # for a_i in range(859):
        #     critic_in.append(list(zip(torch.stack([obs[i] for i in cloest[a_i]]),torch.stack([acs[i] for i in cloest[a_i]]))))
        # critic_rets = [self.critic(o, return_all_q=True) for o in critic_in]
        batch_size = next_obs[0].shape[0]
        samp_acs = samp_acs.view(self.nagents,-1,self.a_dim)
        corrent_5obs = []
        corrent_5acs = []
        tmp_obs = torch.tensor(np.zeros(next_obs[0].shape), dtype=torch.float32, device=next_obs[0].device)
        tmp_acs = torch.tensor(np.zeros(samp_acs[0].shape), dtype=torch.float32, device=next_obs[0].device)
        mask_obs = []
        for a_i in range(self.nagents):
            corrent_5obs.append(torch.stack([obs[i] if i != -1 else tmp_obs for i in cloest[a_i]]))
            corrent_5acs.append(torch.stack([samp_acs[i] if i != -1 else tmp_acs for i in cloest[a_i]]))
            mask_obs.append([mask[a_i] for _  in range(batch_size)])
        # for a_i in range(self.nagents):
        #     corrent_5obs.append(torch.stack([obs[i] if i == cloest[0] else tmp_obs for i in cloest[a_i]]))
        #     corrent_5acs.append(torch.stack([samp_acs[i] if i == cloest[0] else tmp_acs for i in cloest[a_i]]))
        
        mask_obs = torch.ByteTensor(mask_obs).bool().to(next_obs[0].device).view(-1,1,len(mask[0]))

        critic_in = list(zip(torch.cat(corrent_5obs,dim = 1), torch.cat(corrent_5acs,dim = 1)))
        critic_ret = self.critic(critic_in, return_all_q=True, mask=mask_obs)

        q, all_q = critic_ret
        v = (all_q * probs).sum(dim=1, keepdim=True)
        pol_target = q - v
        if soft:
            pol_loss = (log_pis * (log_pis / self.reward_scale - pol_target).detach()).mean()
        else:
            pol_loss = (log_pis * (-pol_target).detach()).mean()
        
        # if regularization
        #pol_loss += 1e-3 * pol_regs[0]  # policy regularization

        # for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs,
        #                                                     all_log_pis, all_pol_regs,
        #                                                     critic_rets):
        #     curr_agent = self.agents[a_i]
        #     v = (all_q * probs).sum(dim=1, keepdim=True)
        #     pol_target = q - v
        #     if soft:
        #         pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
        #     else:
        #         pol_loss = (log_pi * (-pol_target).detach()).mean()
        #     for reg in pol_regs:
        #         pol_loss += 1e-3 * reg  # policy regularization
        #     # don't want critic to accumulate gradients from policy loss
        disable_gradients(self.critic)
        pol_loss.backward()
        enable_gradients(self.critic)

        #grad_norm = torch.nn.utils.clip_grad_norm(self.central_agent.policy.parameters(), 0.5)

        #     for local_param, global_param in zip(curr_agent.policy.parameters(), self.central_agent.policy.parameters()):
        #         if global_param._grad is not None:
        #             global_param._grad = local_param.grad + global_param._grad
        #         else:
        #             global_param._grad = local_param.grad
        #     #curr_agent.policy_optimizer.step()
        #     #self.central_agent.policy_optimizer.step()
        #     curr_agent.policy_optimizer.zero_grad()
        #     #self.central_agent.policy_optimizer.zero_grad()

        
        # for global_param in self.central_agent.policy.parameters():

        #     global_param._grad = global_param._grad / self.nagents
        self.central_agent.policy_optimizer.step()
        self.central_agent.policy_optimizer.zero_grad()
        

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        # for a in self.agents:
        #     soft_update(a.target_policy, a.policy, self.tau)
        soft_update(self.central_agent.target_policy, self.central_agent.policy, self.tau)
        #self.init_from_central_agent()

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        self.central_agent.policy.train()
        self.central_agent.target_policy.train()
        
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.central_agent.policy = fn(self.central_agent.policy)
            #self.central_agent.policy_optimizer = fn(self.central_agent.policy_optimizer)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            #self.critic_optimizer = fn(self.critic_optimizer)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            self.central_agent.target_policy = fn(self.central_agent.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        self.central_agent.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            self.central_agent.policy = fn(self.central_agent.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'central_agent': self.central_agent.get_params(),
                     'central_agent_optimizer': self.central_agent.policy_optimizer.state_dict(),
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, n_agent,s_dim,a_dim, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, dueling=False, norm_in=False,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        # agent_init_params = []
        # sa_size = []
        # for acsp, obsp in zip(env.action_space,
        #                       env.observation_space):
        #     agent_init_params.append({'num_in_pol': obsp.shape[0],
        #                               'num_out_pol': acsp.n})
        #     sa_size.append((obsp.shape[0], acsp.n))
        agent_init_params = [{'num_in_pol': s_dim,
                                       'num_out_pol': a_dim} for i in range(n_agent)]
        sa_size = [(s_dim,a_dim) for i in range(n_agent)]

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size,
                     'dueling': dueling,
                     'norm_in' : norm_in}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.central_agent.load_params(save_dict['central_agent'])
        instance.central_agent.policy_optimizer.load_state_dict(save_dict['central_agent_optimizer'])
        # for state in instance.central_agent.policy_optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.cuda()
        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
            # for state in instance.critic_optimizer.state.values():
            #     for k, v in state.items():
            #         if torch.is_tensor(v):
            #             state[k] = v.cuda()
        return instance

    
    def init_from_central_agent(self):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        for a in self.agents:
            a.load_params(self.central_agent.get_params())

class AttentionSAC_Twin(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)
        self.s_dim = sa_size[0][0]
        self.a_dim = sa_size[0][1]
        self.central_agent = AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **agent_init_params[0])
                                    
        # self.agents = [AttentionAgent(lr=pi_lr,
        #                               hidden_dim=pol_hidden_dim,
        #                               **params)
        #                  for params in agent_init_params]
        self.critic1 = AttentionCritic(sa_size[:5], hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        
        self.critic1_old = deepcopy(self.critic1)

        self.critic2 = AttentionCritic(sa_size[:5], hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        
        self.critic2_old = deepcopy(self.critic2)
        
        #hard_update(self.target_critic, self.critic)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        #self.init_from_central_agent()
        # self.grad = AttentionAgent(lr=pi_lr,
        #                               hidden_dim=pol_hidden_dim,
        #                               **agent_init_params[0])
    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        # return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
        #                                                        observations)]

        #before = [a.step(obs) for a, obs in zip(self.agents, observations)]

        obs = torch.cat(observations)
        act =self.central_agent.step(obs, explore=explore)
        act = act.view(self.nagents,-1,self.a_dim)
        return [act[i] for i in range(self.nagents)]


    def update_critic(self, sample,cloest, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """

        obs, acs, rews, next_obs, dones = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        pi = self.central_agent.target_policy

        agent_num = len(next_obs)
        batch_size = next_obs[0].shape[0]

        with torch.no_grad():
            tmp = torch.stack(next_obs).view(-1, self.s_dim)
            next_acs, next_log_pis = pi(tmp, return_log_pi=True)
            next_acs = next_acs.view(agent_num, batch_size, -1)

        next_5obs = []
        next_5acs = []
        for a_i in range(self.nagents):
            next_5obs.append(torch.stack([next_obs[i] for i in cloest[a_i]]))
            next_5acs.append(torch.stack([next_acs[i] for i in cloest[a_i]]))
        #5 * (threads*agents) *54
        trgt_critic_in = list(zip(torch.cat(next_5obs,dim = 1), torch.cat(next_5acs,dim = 1)))

        corrent_5obs = []
        corrent_5acs = []
        for a_i in range(self.nagents):
            corrent_5obs.append(torch.stack([obs[i] for i in cloest[a_i]]))
            corrent_5acs.append(torch.stack([acs[i] for i in cloest[a_i]]))
        critic_in = list(zip(torch.cat(corrent_5obs,dim = 1), torch.cat(corrent_5acs,dim = 1)))
        #trgt_critic_in = list(zip(next_obs, next_acs))
        #critic_in = list(zip(obs, acs))

        with torch.no_grad():
            next_qs2 = self.critic2_old(trgt_critic_in)
            next_qs1 = self.critic1_old(trgt_critic_in)

            next_qs = torch.min(next_qs1, next_qs2)

        critic_rets1 = self.critic1(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        
        critic_rets2 = self.critic2(critic_in, regularize=True,
                                   logger=logger, niter=self.niter)
        
        q_loss1 = 0
        q_loss2 = 0
        pq1, regs1 = critic_rets1
        pq2, regs2 = critic_rets2
        target_q = (torch.cat(rews).view(-1, 1) + self.gamma * next_qs *  (1 - torch.cat(dones).view(-1, 1)))
        if soft:
            target_q -= next_log_pis / self.reward_scale
        q_loss1 += MSELoss(pq1, target_q.detach())
        q_loss2 += MSELoss(pq2, target_q.detach())

        ''' if regularization
        for reg in regs:
            q_loss += reg  # regularizing attention
        '''

        q_loss1.backward()
        q_loss2.backward()
        # original maac 10 * self.nagents
        #grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.nagents)
        
        self.critic1_optimizer.zero_grad()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        self.critic2_optimizer.step()

        if logger is not None:
            logger.add_scalar('losses/q1_loss', q_loss1, self.niter)
            logger.add_scalar('losses/q2_loss', q_loss2, self.niter)
            #logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample,cloest, soft=True, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample
        # samp_acs = []
        # all_probs = []
        # all_log_pis = []
        # all_pol_regs = []

        # for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
        #     curr_ac, probs, log_pi, pol_regs, ent = pi(
        #         ob, return_all_probs=True, return_log_pi=True,
        #         regularize=True, return_entropy=True)
        #     logger.add_scalar('policy_entropy' %ent,
        #                       self.niter)
        #     samp_acs.append(curr_ac)
        #     all_probs.append(probs)
        #     all_log_pis.append(log_pi)
        #     all_pol_regs.append(pol_regs)
        obs1 = torch.cat(obs)
        samp_acs,probs,log_pis,pol_regs =  self.central_agent.policy(
                obs1, return_all_probs=True, return_log_pi=True,
                regularize=True)

        # critic_in = []
        # for a_i in range(859):
        #     critic_in.append(list(zip(torch.stack([obs[i] for i in cloest[a_i]]),torch.stack([acs[i] for i in cloest[a_i]]))))
        # critic_rets = [self.critic(o, return_all_q=True) for o in critic_in]

        corrent_5obs = []
        corrent_5acs = []
        
        for a_i in range(self.nagents):
            corrent_5obs.append(torch.stack([obs[i] for i in cloest[a_i]]))
            corrent_5acs.append(torch.stack([samp_acs.view(self.nagents,-1,self.a_dim)[i] for i in cloest[a_i]]))
        critic_in = list(zip(torch.cat(corrent_5obs,dim = 1), torch.cat(corrent_5acs,dim = 1)))
        
        critic1_ret = self.critic1(critic_in, return_all_q=True)
        critic2_ret = self.critic2(critic_in, return_all_q=True)

        q1, all_q1 = critic1_ret
        q2, all_q2 = critic2_ret

        pol_loss = (-torch.min(q1,q2) + log_pis / self.reward_scale).mean()

        #v = (all_q * probs).sum(dim=1, keepdim=True)
        #pol_target = q - v

        #if soft:
        #    pol_loss = (log_pis * (log_pis / self.reward_scale - pol_target).detach()).mean()
        #else:
        #    pol_loss = (log_pis * (-pol_target).detach()).mean()
        
        # if regularization
        #pol_loss += 1e-3 * pol_regs[0]  # policy regularization

        # for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs,
        #                                                     all_log_pis, all_pol_regs,
        #                                                     critic_rets):
        #     curr_agent = self.agents[a_i]
        #     v = (all_q * probs).sum(dim=1, keepdim=True)
        #     pol_target = q - v
        #     if soft:
        #         pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
        #     else:
        #         pol_loss = (log_pi * (-pol_target).detach()).mean()
        #     for reg in pol_regs:
        #         pol_loss += 1e-3 * reg  # policy regularization
        #     # don't want critic to accumulate gradients from policy loss
        disable_gradients(self.critic1)
        disable_gradients(self.critic2)
        pol_loss.backward()
        enable_gradients(self.critic1)
        enable_gradients(self.critic2)


        #grad_norm = torch.nn.utils.clip_grad_norm(self.central_agent.policy.parameters(), 0.5)

        #     for local_param, global_param in zip(curr_agent.policy.parameters(), self.central_agent.policy.parameters()):
        #         if global_param._grad is not None:
        #             global_param._grad = local_param.grad + global_param._grad
        #         else:
        #             global_param._grad = local_param.grad
        #     #curr_agent.policy_optimizer.step()
        #     #self.central_agent.policy_optimizer.step()
        #     curr_agent.policy_optimizer.zero_grad()
        #     #self.central_agent.policy_optimizer.zero_grad()

        
        # for global_param in self.central_agent.policy.parameters():

        #     global_param._grad = global_param._grad / self.nagents
        self.central_agent.policy_optimizer.zero_grad()
        self.central_agent.policy_optimizer.step()
        
    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.critic1_old, self.critic1, self.tau)
        soft_update(self.critic2_old, self.critic2, self.tau)
        # for a in self.agents:
        #     soft_update(a.target_policy, a.policy, self.tau)
        soft_update(self.central_agent.target_policy, self.central_agent.policy, self.tau)
        #self.init_from_central_agent()

    def prep_training(self, device='gpu'):
        self.critic1.train()
        self.critic2.train()
        self.central_agent.policy.train()
        self.central_agent.target_policy.train()
        
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.central_agent.policy = fn(self.central_agent.policy)
            #self.central_agent.policy_optimizer = fn(self.central_agent.policy_optimizer)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic1 = fn(self.critic1)
            self.critic2 = fn(self.critic2)
            #self.critic_optimizer = fn(self.critic_optimizer)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            self.central_agent.target_policy = fn(self.central_agent.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.critic1_old = fn(self.critic1_old)
            self.critic2_old = fn(self.critic2_old)
            #self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        self.central_agent.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            self.central_agent.policy = fn(self.central_agent.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'central_agent': self.central_agent.get_params(),
                     'central_agent_optimizer': self.central_agent.policy_optimizer.state_dict(),
                     'critic1_params': {'critic': self.critic1.state_dict(),
                                       'critic1_old': self.critic1_old.state_dict(),
                                       'critic_optimizer': self.critic1_optimizer.state_dict()},
                     'critic2_params': {'critic': self.critic2.state_dict(),
                                       'critic1_old': self.critic2_old.state_dict(),
                                       'critic_optimizer': self.critic2_optimizer.state_dict()}
                                       }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, s_dim,a_dim, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        # agent_init_params = []
        # sa_size = []
        # for acsp, obsp in zip(env.action_space,
        #                       env.observation_space):
        #     agent_init_params.append({'num_in_pol': obsp.shape[0],
        #                               'num_out_pol': acsp.n})
        #     sa_size.append((obsp.shape[0], acsp.n))
        agent_init_params = [{'num_in_pol': s_dim,
                                       'num_out_pol': a_dim} for i in range(self.nagents)]
        sa_size = [(s_dim,a_dim) for i in range(self.nagents)]

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.central_agent.load_params(save_dict['central_agent'])
        instance.central_agent.policy_optimizer.load_state_dict(save_dict['central_agent_optimizer'])
        for state in instance.central_agent.policy_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        if load_critic:
            critic_params = save_dict['critic1_params']
            instance.critic1.load_state_dict(critic_params['critic'])
            instance.critic1_old.load_state_dict(critic_params['critic_old'])
            instance.critic1_optimizer.load_state_dict(critic_params['critic_optimizer'])
            for state in instance.critic1_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        
            critic_params = save_dict['critic2_params']
            instance.critic2.load_state_dict(critic_params['critic'])
            instance.critic2_old.load_state_dict(critic_params['critic_old'])
            instance.critic2_optimizer.load_state_dict(critic_params['critic_optimizer'])
            for state in instance.critic2_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        return instance

    
    def init_from_central_agent(self):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        for a in self.agents:
            a.load_params(self.central_agent.get_params())
