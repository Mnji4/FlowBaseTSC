import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
from pathlib import Path
from utils.buffer import ReplayBuffer,ReplayBufferTime
# from algorithms.attention_sac1 import AttentionSAC
#from algorithms.attention_ppo1 import AttentionPPO
from algorithms.distral import Distral, PairAgent
import json
from utils.ma_env_time import MaEnv,make_env,make_parallel_env
from utils.misc import onehot_from_logits, categorical_sample, epsilon_greedy
from tqdm import trange
import cProfile
from utils.grouper import Grouper
def run(config, start = 0):
    #count_cloest()
    best_rew = 0

    
    model_dir = Path('models') / config.method / config.env_name
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
    str_run = 'run%i' % run_num
    run_dir = model_dir / str_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)

    with open(os.path.join(run_dir, 'param.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)


    torch.manual_seed(run_num)
    np.random.seed(run_num)
    config_file = f"./config/config_{config.env_name}.json"
    env = make_parallel_env(config_file, config.n_rollout_threads, run_num)
    g = Grouper(env.env)
    g.grouping_ns()
    replay_buffer_inter = ReplayBufferTime(config.buffer_length, 1,
                                [env.observation_space.shape[1] for i in range(1)],
                                [env.action_space.n for i in range(1)])
    replay_buffer_traj = ReplayBufferTime(config.buffer_length, 1,
                                [env.observation_space.shape[1] for i in range(1)],
                                [env.action_space.n for i in range(1)])
    env.env.env.traj_buffer = replay_buffer_traj
    env.env.env.traj_gamma = 0.99

    if config.load_model:
        filename = Path(config.model_path)
        model = Distral.init_from_save(filename, load_critic=True)
    else:
        model_inter = Distral.init_from_env(env,  s_dim=env.observation_space.shape[1],
                                        a_dim= env.action_space.n,
                                        n_agent= 1,
                                        tau=config.tau,
                                        pi_lr=config.pi_lr,
                                        q_lr=config.q_lr,
                                        gamma=config.gamma,
                                        pol_hidden_dim=config.pol_hidden_dim,
                                        critic_hidden_dim=config.critic_hidden_dim)

        model_pair = PairAgent.init_from_env(env,  s_dim=env.observation_space.shape[1],
                                        a_dim= env.action_space.n,
                                        n_agent= 1,
                                        tau=config.tau,
                                        pi_lr=config.pi_lr,
                                        q_lr=config.q_lr,
                                        gamma=config.gamma,
                                        pol_hidden_dim=config.pol_hidden_dim,
                                        critic_hidden_dim=config.critic_hidden_dim)
    #filename = Path('other/MAACcbe/models/CBE/colight_sac_round2_pressure_r/run2/incremental/model_ep11.pt')                                  
    #model = AttentionSAC.init_from_save(filename, load_critic=True)



    t = 0
    for i in range(config.n_rollout_threads):
        if config.use_gpu:
            model_inter.prep_rollouts(device='cuda')
            model_pair.prep_rollouts(device='cuda')
        else:
            model_inter.prep_rollouts(device='cpu')
            model_pair.prep_rollouts(device='cpu')
    for ep_i in range(start, config.n_episodes, config.n_rollout_threads):
        coef = 0.5
        if ep_i % config.test_interval < config.n_rollout_threads and start != ep_i:
            print('testing policies')
            obs = env.reset()
            coef = 1 - min(ep_i,50) /50
            for test_t in range(0,3600,env.seconds_per_step):
                
                # torch_obs = [torch.Tensor(obs).cuda() for ob in obs]
                torch_obs = torch.Tensor(obs).cuda()
                #[thread,agent,act]
                all_q_inter = model_inter.step(torch_obs[0], explore=True, return_all_q=True)
                
                paired_obs, unpaired_obs = g.to_group_obs(obs)
                all_q_pair = model_pair.step(torch.Tensor(paired_obs).cuda()[0], explore=True, return_all_q=True)[0]
                parsed_q_pair = g.parse_group_action([],all_q_pair)
                # rearrange actions to be per environment
                #[thread,agent,act]
                
                all_q = all_q_inter*coef+parsed_q_pair*(1-coef)
                act = onehot_from_logits(all_q)
                # rearranges  actionto be per environment
                #[thread,agent,act]
                actions = [act.cpu().numpy()]
                next_obs, rewards, dones, infos = env.step(actions)
                obs = next_obs


        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()

        for et_i in range(0,config.episode_length,env.seconds_per_step):
            # torch_obs = [torch.Tensor(obs).cuda() for ob in obs]
            torch_obs = torch.Tensor(obs).cuda()
            #[thread,agent,act]
            with torch.no_grad():
                all_q_inter = model_inter.step(torch_obs[0], explore=True, return_all_q=True)
            
            paired_obs, unpaired_obs = g.to_group_obs(obs)
            with torch.no_grad():
                all_q_pair = model_pair.step(torch.Tensor(paired_obs.squeeze(1)).cuda(), explore=True, return_all_q=True)
            
            parsed_q_pair = g.parse_group_action(all_q_pair.cpu().numpy(),None)
            # rearrange actions to be per environment
            #[thread,agent,act]
            
            all_q = all_q_inter*coef+torch.tensor(parsed_q_pair).cuda()*(1-coef)
            probs = F.softmax(all_q, dim=1)
            int_acs, act = categorical_sample(probs, use_cuda=config.use_gpu)
            actions = [act.cpu().numpy()]
            next_obs, rewards, dones, infos = env.step(actions)

            #分配各个option的经历

            for th in range(config.n_rollout_threads):
                
                _obs = np.expand_dims(obs[th],1)
                _actions = np.expand_dims(actions[th],0)
                _rewards = np.expand_dims(rewards[th],1)
                _times = np.full_like(_rewards,(et_i+1)/env.seconds_per_step)
                _next_obs = np.expand_dims(next_obs[th],1)
                _dones = np.expand_dims(dones[th],1)
                replay_buffer_inter.push(_obs, _actions, _rewards, _next_obs, _dones,_times)

            
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer_traj) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                for i in range(config.n_rollout_threads):

                    if config.use_gpu:
                        model_inter.prep_training(device='cuda')
                        model_traj.prep_training(device='cuda')
                    else:
                        model_inter.prep_training(device='cpu')
                        model_traj.prep_training(device='cpu')
                    for u_i in range(config.num_updates):
                        sample = replay_buffer_inter.sample(config.batch_size,
                                                    to_gpu=config.use_gpu)
                        model_inter.optimize_model(sample)#model[0]
                        model_inter.update_all_targets()
                    for u_i in range(config.num_updates):
                        sample = replay_buffer_traj.sample(config.batch_size,
                                                    to_gpu=config.use_gpu)
                        model_traj.optimize_model(sample)#model[0]
                        model_traj.update_all_targets()
                    if config.use_gpu:
                        model_inter.prep_rollouts(device='cuda')
                        model_traj.prep_rollouts(device='cuda')
                    else:
                        model_inter.prep_rollouts(device='cpu')
                        model_traj.prep_rollouts(device='cpu')
                

        # if ep_i % config.save_interval < config.n_rollout_threads:
        #     os.makedirs(run_dir , exist_ok=True)
        #     model[0].save(run_dir / f'model_ep{ep_i + 1}.pt')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",default='Mix', help="Name of environment")
    parser.add_argument("--env_name",default='jinan')
    parser.add_argument("--dqn", default=1, type=int)
    parser.add_argument("--n_agent", default=1, type=int)
    parser.add_argument("--test_interval", default=10, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e7), type=int)
    parser.add_argument("--n_episodes", default=201, type=int)
    parser.add_argument("--episode_length", default=3600, type=int)
    parser.add_argument("--steps_per_update", default=10, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=511, type=int,
                        help="Batch size for training")    
    parser.add_argument("--meta_batch_size",
                        default=256, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=20, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.0002, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--use_gpu", default=True, action='store_true')

    parser.add_argument("--log_num",default=0, type=int)
    parser.add_argument("--load_model", default=False, type=bool)
    parser.add_argument("--model_path", default='models/Traj/run23/model_ep181.pt')
    config = parser.parse_args()
    run(config, 0)
    # cProfile.run("")
