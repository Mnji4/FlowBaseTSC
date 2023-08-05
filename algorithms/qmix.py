import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import SingleCritic 
from utils.mixer import QMixNet
from utils.misc import onehot_from_logits, categorical_sample
import numpy as np
MSELoss = torch.nn.MSELoss()

class Qmix(object):
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


        # self.agents = [AttentionAgent(lr=pi_lr,
        #                               hidden_dim=pol_hidden_dim,
        #                               **params)
        #                  for params in agent_init_params]
        self.critic = SingleCritic(sa_size[0], hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads, norm_in = False)
        self.target_critic = SingleCritic(sa_size[0], hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads, norm_in = False)
        hard_update(self.target_critic, self.critic)

        #self.mixer_device = 'cuda:0'
        state_dim = self.s_dim + self.a_dim

        self.action_mask = [0] * self.s_dim*self.nagents
        for i in range(self.nagents):
            for j in range(0,self.a_dim):
                self.action_mask[i*self.s_dim+j] = 1

        self.qmixer = QMixNet(n_agents=self.nagents, obs_dim = state_dim)
        self.target_qmixer = QMixNet(n_agents=self.nagents, obs_dim = state_dim)
        hard_update(self.qmixer, self.target_qmixer)

        #self.qmixer_optimizer = Adam(self.qmixer.parameters(), lr=q_lr,weight_decay=1e-3)
        #self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)
        self.critic_optimizer = Adam(list(self.critic.parameters()) + list(self.qmixer.parameters()), lr=q_lr,
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


    def step(self, obs, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """

        corrent_5obs = []
        mask_obs = []
        batch_size = obs[0].shape[0]
        tmp_obs = torch.tensor(np.zeros(obs[0].shape), dtype=torch.float32, device=obs[0].device)
        for a_i in range(self.nagents):
            corrent_5obs.append(torch.stack([obs[i] if i == 0 else tmp_obs for i in range(5)]))
        critic_in = list(torch.cat(corrent_5obs,dim = 1))
        
        with torch.no_grad():
            act = self.critic(critic_in, mask=mask_obs, return_all_q=False, return_act=True, explore=explore)

        act = act.view(self.nagents,-1,self.a_dim)

        return [act[i] for i in range(self.nagents)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """

        obs, acs, rews, next_obs, dones = sample

        #pi = self.central_agent.target_policy

        agent_num = len(next_obs)
        batch_size = next_obs[0].shape[0]


        
        #5 * (threads*agents) *54
        #trgt_critic_in = list(torch.stack(next_5obs).permute(1,0,2,3).reshape(5, agent_num*batch_size, -1))
        trgt_critic_in = next_obs


        
        #critic_in = list(torch.stack(corrent_5obs).permute(1,0,2,3).reshape(5, agent_num*batch_size, -1))
        critic_in = list(torch.cat((obs,acs),dim = 1))

        with torch.no_grad():
            next_q = self.target_critic(trgt_critic_in, mask=next_mask_obs, return_all_q=False, return_q=True)
        
        critic_rets = self.critic(critic_in, regularize=True,return_all_q=True,
                                  logger=logger, niter=self.niter, mask=mask_obs)
        q_loss = 0
        
        all_q, regs = critic_rets

        cur_action = corrent_5acs.max(dim=1, keepdim=True)[1]
        cur_q = all_q.gather(1, cur_action)

        #local q loss
        target_q = (torch.cat(rews).view(-1, 1) + self.gamma * next_q *  (1 - torch.cat(dones).view(-1, 1)))

        local_q_loss = MSELoss(cur_q, target_q.detach())
        # mix netwoek
        # reshape q values (batch, nagents, 1)

        cur_q = cur_q.view(agent_num, 1, batch_size).permute(2,1,0)
        next_q = next_q.view(agent_num, 1, batch_size).permute(2,1,0)

        
        def count_mix_obs(obs,road_mask):
            states = []

            for i in range(self.nagents):
                for j in range(4):
                    if road_mask[i][j] == 0:
                        # states.append(torch.cat((obs[:,self.s_dim*i + self.a_dim + 3*j:self.s_dim*i + 
                        # self.a_dim + 3*j + 3],obs[:,self.s_dim*i + self.a_dim+12 + 3*j:self.s_dim*i + self.a_dim+12 + 3*j + 3]),dim = 1))
                        states.append(torch.cat((obs[:,self.s_dim*i  + 3*j:self.s_dim*i + 
                         3*j + 3],obs[:,self.s_dim*i + 12 + 3*j:self.s_dim*i + 12 + 3*j + 3]),dim = 1))
            batch_state = torch.cat(states,1)
            acts = []
            for state in obs:
                acts.append(torch.masked_select(state,mask=torch.ByteTensor(self.action_mask)))
            batch_act = torch.stack(acts)
            res = torch.cat((batch_act,batch_state),dim = 1)
            return res

        flat_obs = torch.stack(obs).permute(1,0,2).reshape(batch_size,-1)
        flat_next_obs = torch.stack(next_obs).permute(1,0,2).reshape(batch_size,-1)

        mix_obs = count_mix_obs(flat_obs,road_mask)
        mix_next_obs = count_mix_obs(flat_next_obs,road_mask)
        mix_cur_q = self.qmixer(cur_q, mix_obs)
        with torch.no_grad():
            mix_next_q = self.target_qmixer(next_q, mix_next_obs)
        
        rews = torch.stack(rews).permute(1,0).sum(1).view(-1,1)
        dones = torch.stack(dones).permute(1,0).mean(1).view(-1,1)
        target_q = rews + self.gamma * mix_next_q *  (1 - dones)

        global_q_loss = MSELoss(mix_cur_q, target_q.detach())
        # if self.niter < 1000:
        #     q_loss = local_q_loss
        # else:
        #     q_loss = global_q_loss
        q_loss = global_q_loss
        ''' if regularization
        for reg in regs:
            q_loss += reg  # regularizing attention
        '''

        # original maac 10 * self.nagents
        # grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.nagents)
        #self.qmixer_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        #self.qmixer_optimizer.step()
        self.critic_optimizer.step()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('losses/local_q_loss', local_q_loss, self.niter)
            logger.add_scalar('losses/global_q_loss', global_q_loss, self.niter)
            #logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_qmixer, self.qmixer, self.tau)
        # for a in self.agents:
        #     soft_update(a.target_policy, a.policy, self.tau)
        #self.init_from_central_agent()

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        self.qmixer.train()
        self.target_qmixer.train()
        
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.qmixer = fn(self.qmixer)
            #self.critic_optimizer = fn(self.critic_optimizer)
            self.critic_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.target_qmixer = fn(self.target_qmixer)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        #self.central_agent.policy.eval()
        self.critic.eval()
        self.qmixer.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            #self.qmixer = fn(self.qmixer)
            #self.critic_optimizer = fn(self.critic_optimizer)
            self.critic_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict(),
                                       'qmixer': self.qmixer.state_dict(),
                                       'target_qmixer':self.target_qmixer.state_dict()}
                    }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, s_dim,a_dim, n_agent ,gamma=0.95, tau=0.01,
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
                                       'num_out_pol': a_dim} for i in range(n_agent)]
        sa_size = [(s_dim,a_dim) for i in range(n_agent)]

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
        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.qmixer.load_state_dict(critic_params['qmixer'])
            instance.target_qmixer.load_state_dict(critic_params['target_qmixer'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
            for state in instance.critic_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        return instance
