import argparse
import torch
import os
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBufferTime
# from algorithms.attention_sac1 import AttentionSAC
#from algorithms.attention_ppo1 import AttentionPPO
from algorithms.distral import Distral
import json
# from utils.ma_env_time import make_parallel_env
from env.myenv import make_parallel_env

def run(config, start = 0):
    #count_cloest()
    best_rew = 0

    
    model_dir = Path('models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)

    with open(os.path.join(run_dir, 'param.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)

    logger = SummaryWriter(str(log_dir))


    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env('./config/config_jinan.json', config.n_rollout_threads, 42)
    model = []
    replay_buffer = []

    config.s_dim = config.s_dim + config.a_dim

    for _ in range(config.n_rollout_threads):

        model_ = Distral.init_from_env(env,  s_dim=config.s_dim,
                                        a_dim= config.a_dim,
                                        n_agent= 1,
                                        tau=config.tau,
                                        pi_lr=config.pi_lr,
                                        q_lr=config.q_lr,
                                        gamma=config.gamma,
                                        pol_hidden_dim=config.pol_hidden_dim,
                                        critic_hidden_dim=config.critic_hidden_dim,
                                        attend_heads=config.attend_heads,
                                        reward_scale=config.reward_scale)
        model.append(model_)

        replay_buffer_ = ReplayBufferTime(360000//config.interval_length, 1,
                                [config.s_dim for i in range(1)],
                                [1 for i in range(1)])
        replay_buffer.append(replay_buffer_)
        
    if config.load_model:
        for i in range(2):
            filename = Path(config.model_path + 'model%i_ep121.pt' %(i))
            model_ = Distral.init_from_save(filename, load_critic=True)
            model.append(model_)


    t = 0
    

    batchsizes = [0,0]
    for ep_i in range(start, config.n_episodes, config.n_rollout_threads):
        if 0 and ep_i % config.test_interval < config.n_rollout_threads:
            print('testing central policy')
            obs = env.reset()
            for test_t in range(0,3600,config.interval_length):
                torch_obs = [torch.Tensor(ob) for ob in obs]
                # get actions as torch Variables
                #[thread,agent,act]
                actions = [model[0].step([torch_obs[i]], explore=False) for i in range(config.n_rollout_threads)]
                # rearrange actions to be per environment
                #[thread,agent,act]
                actions = [a.numpy() for a in actions]
                next_obs, rewards, dones, infos = env.step(actions)
                obs = next_obs



            print('testing policies')
            obs = env.reset()
            for test_t in range(0,3600,config.interval_length):
                torch_obs = [torch.Tensor(ob) for ob in obs]
                # get actions as torch Variables
                #[thread,agent,act]
                actions = [model[i].step([torch_obs[i]], explore=False) for i in range(config.n_rollout_threads)]

                # rearrange actions to be per environment
                #[thread,agent,act]
                actions = [a.numpy() for a in actions]
                next_obs, rewards, dones, infos = env.step(actions)
                obs = next_obs


        if ep_i >100:
            for i in range(2):
                model[i].critic.epsilon = 0.99
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()

        ep_rew_mean = np.zeros(config.n_rollout_threads)
        rewards = np.zeros((config.n_rollout_threads,config.n_agent))
        
        for et_i in range(0,config.episode_length,config.interval_length):
            #[agent,thread,obs]
            # torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
            #                       requires_grad=False)
            #              for i in range(config.n_agent)]
            #[thread,agent,obs]
            torch_obs = [torch.Tensor(ob) for ob in obs]
            # get actions as torch Variables
            #[thread,agent,act]
            actions = [model[i].step([torch_obs[i]], explore=True) for i in range(config.n_rollout_threads)]

            # rearrange actions to be per environment
            #[thread,agent,act]
            next_obs, rewards, dones, infos = env.step(actions)
            r_step = np.array([sum(o) for o in rewards])
            ep_rew_mean = (ep_rew_mean * et_i + r_step*config.interval_length)/(et_i+config.interval_length)


            if et_i%30000 == 3600:
                #print([(np.argmax(actions[0][i])+1) for i in range(22)])
                print('ep %i step %i' %(ep_i,et_i))
                print(r_step)
                print(ep_rew_mean)
            #分配各个option的经历

            for th in range(config.n_rollout_threads):
                
                _obs = np.expand_dims(obs[th],1)
                _actions = np.expand_dims(actions[th],1)
                _rewards = np.expand_dims(rewards[th],1)
                _times = np.full_like(_rewards,(et_i+1)/config.interval_length)
                _next_obs = np.expand_dims(next_obs[th],1)
                _dones = np.expand_dims(dones[th],1)
                replay_buffer[th].push(_obs, _actions, _rewards, _next_obs, _dones,_times)

            
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer[0]) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                for i in range(config.n_rollout_threads):

                    if config.use_gpu:
                        model[i].prep_training(device='gpu')
                    else:
                        model[i].prep_training(device='cpu')
                    for u_i in range(config.num_updates):
                        sample = replay_buffer[i].sample(config.batch_size,
                                                    to_gpu=config.use_gpu)
                        model[i].optimize_model(sample, model[0].critic)#model[0]
                        model[i].update_all_targets()
                    model[i].prep_rollouts(device='cpu')
                






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",default='Distral', help="Name of environment")
    parser.add_argument("--model_name",default='test',
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--dqn", default=1, type=int)
    parser.add_argument("--s_dim", default=152, type=int)
    parser.add_argument("--a_dim", default=8, type=int)
    parser.add_argument("--n_agent", default=1, type=int)
    parser.add_argument("--interval_length", default=1, type=int)
    parser.add_argument("--test_interval", default=10, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--n_episodes", default=200, type=int)
    parser.add_argument("--episode_length", default=3600, type=int)
    parser.add_argument("--steps_per_update", default=30, type=int)
    parser.add_argument("--num_updates", default=2, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=128, type=int,
                        help="Batch size for training")    
    parser.add_argument("--meta_batch_size",
                        default=256, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=40, type=int)
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
    parser.add_argument("--load_model", default=False, type=bool)
    parser.add_argument("--model_path", default='/start/manhattan/models/Distral/test/run113/')
    config = parser.parse_args()

    run(config, 0)
