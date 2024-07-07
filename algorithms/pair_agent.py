import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import SingleCritic 
from utils.misc import onehot_from_logits, categorical_sample
import numpy as np
MSELoss = torch.nn.MSELoss()


class PairAgent(object):
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.001,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):


        self.nagents = len(sa_size)
        self.s_dim = sa_size[0][0]
        self.a_dim = sa_size[0][1]


        # self.agents = [AttentionAgent(lr=pi_lr,
        #                               hidden_dim=pol_hidden_dim,
        #                               **params)
        #                  for params in agent_init_params]
        self.critic = SingleCritic(sa_size[:5], hidden_dim=critic_hidden_dim,
                                       norm_in = False)
        self.target_critic = SingleCritic(sa_size[:5], hidden_dim=critic_hidden_dim,
                                              norm_in = False)
        hard_update(self.target_critic, self.critic)



        #self.qmixer_optimizer = Adam(self.qmixer.parameters(), lr=q_lr,weight_decay=1e-3)
        #self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        #self.init_from_central_agent()
        # self.grad = AttentionAgent(lr=pi_lr,
        #                               hidden_dim=pol_hidden_dim,
        #                               **agent_init_params[0])


    def step(self, obs, explore=False, return_all_q=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """

        # corrent_5obs = []
        # mask_obs = []
        # batch_size = obs[0].shape[0]
        # tmp_obs = torch.tensor(np.zeros(obs[0].shape), dtype=torch.float32, device=obs[0].device)
        # for a_i in range(self.nagents):
        #     corrent_5obs.append(torch.stack([obs[i] if i == 0 else tmp_obs for i in range(5)]))
        # critic_in = list(torch.cat(corrent_5obs,dim = 1))
        critic_in = obs
        with torch.no_grad():
            act = self.critic(critic_in, mask=[], return_all_q=False, return_act=True, explore=explore)


        act = act.view(self.nagents,-1,self.a_dim)

        # probs = self.critic(critic_in, mask=mask_obs, return_all_q=False, return_probs=True, explore=explore)        
        # probs = probs.view(self.nagents,-1,self.a_dim)

        all_q = self.critic(critic_in, mask=[], return_all_q=True, return_logits=False, explore=explore)        
        all_q = all_q.view(self.nagents,-1,self.a_dim)
        if return_all_q:
            act = all_q
        return [act[i] for i in range(self.nagents)]
        # return [act[i] for i in range(self.nagents)],[probs[i] for i in range(self.nagents)]
        #return [act[i] for i in range(self.nagents)],[logits[i] for i in range(self.nagents)]


    def optimize_model(self, sample):
        gamma=self.gamma
        obs, acs, rews, next_obs, dones, times = sample

        next_obs = next_obs[0]#torch.Tensor(next_obs[0])
        state_batch = obs[0]#torch.Tensor(obs[0])
        action_batch = acs[0]#torch.Tensor(acs[0])
        reward_batch1 = rews#torch.Tensor(rews[0])
        reward_batch = reward_batch1[0].unsqueeze(1)
        if torch.any(reward_batch==float('-inf')):
            print('reward_batch inf')

        state_action_values = self.critic(state_batch, return_all_q = True).gather(1, action_batch.argmax(dim = 1,keepdim = True))
        
        next_state_values = self.target_critic(next_obs, return_all_q = True).gather(1, action_batch.argmax(dim = 1,keepdim = True))
        
        expected_state_action_values = (next_state_values * gamma)* (1 - dones[0].view(-1,1)) + reward_batch.view(-1,1)
        
        # Compute MSE loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.detach())
        if np.random.randint(1,100)<1:
            print(loss.item())
            print(next_state_values.mean().item(),expected_state_action_values.mean().item())
        # if(np.random.randint(0,100)<3):
        #     print(state_action_values.detach().mean().item(),
        #             expected_state_action_values.detach().mean().item())
        #     print(loss.item())
        # Optimize the model
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        if torch.any(torch.isnan(self.critic.state_encoder.s_enc_fc1.weight)):
            print()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        # for a in self.agents:
        #     soft_update(a.target_policy, a.policy, self.tau)
        #self.init_from_central_agent()

    def prep_training(self, device='cuda'):
        self.critic.train()
        self.target_critic.train()
        
        if device == 'cuda':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            #self.critic_optimizer = fn(self.critic_optimizer)
            self.critic_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        #self.central_agent.policy.eval()
        self.critic.eval()
        if device == 'cuda':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            #self.critic_optimizer = fn(self.critic_optimizer)
            self.critic_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device
    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        device=self.critic_dev
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}
                    }
        torch.save(save_dict, filename)
        self.prep_training(device=device)
    @classmethod
    def init_from_env(cls, env, s_dim,a_dim, n_agent ,gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      pol_hidden_dim=128, critic_hidden_dim=128,
                      **kwargs):

        agent_init_params = [{'num_in_pol': s_dim,
                                       'num_out_pol': a_dim} for i in range(n_agent)]
        sa_size = [(s_dim,a_dim) for i in range(n_agent)]

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
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
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
            for state in instance.critic_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        return instance
