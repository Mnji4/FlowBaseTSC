import argparse
import torch
import os
import numpy as np
from pathlib import Path
from utils.buffer import ReplayBufferTime
# from algorithms.attention_sac1 import AttentionSAC
#from algorithms.attention_ppo1 import AttentionPPO
from algorithms.distral import Distral
import json
from utils.ma_env_time import MaEnv,make_env,make_parallel_env
from tqdm import trange
import cProfile
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
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)

    with open(os.path.join(run_dir, 'param.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)


    torch.manual_seed(run_num)
    np.random.seed(run_num)
    config_file = f"./config/config_{config.env_name}.json"
    env = make_parallel_env(config_file, config.n_rollout_threads, run_num)
    model = []
    replay_buffer = []


    for _ in range(config.n_rollout_threads):

        model_ = Distral.init_from_env(env,  s_dim=env.observation_space.shape[1],
                                        a_dim= env.action_space.n,
                                        n_agent= 1,
                                        tau=config.tau,
                                        pi_lr=config.pi_lr,
                                        q_lr=config.q_lr,
                                        gamma=config.gamma,
                                        pol_hidden_dim=config.pol_hidden_dim,
                                        critic_hidden_dim=config.critic_hidden_dim)
        model.append(model_)

        replay_buffer_ = ReplayBufferTime(360000//env.seconds_per_step, 1,
                                [env.observation_space.shape[1] for i in range(1)],
                                [env.action_space.n for i in range(1)])
        replay_buffer.append(replay_buffer_)
    #filename = Path('other/MAACcbe/models/CBE/colight_sac_round2_pressure_r/run2/incremental/model_ep11.pt')                                  
    #model = AttentionSAC.init_from_save(filename, load_critic=True)

    if config.load_model:
        filename = Path(config.model_path)
        model[0] = Distral.init_from_save(filename, load_critic=True)


    t = 0
    for i in range(config.n_rollout_threads):
        if config.use_gpu:
            model[i].prep_rollouts(device='cuda')
        else:
            model[i].prep_rollouts(device='cpu')
    for ep_i in range(start, config.n_episodes, config.n_rollout_threads):
        if ep_i % config.test_interval < config.n_rollout_threads and start != ep_i:
            print('testing policies')
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


        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()

        for et_i in range(0,config.episode_length,env.seconds_per_step):
            #[agent,thread,obs]
            # torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
            #                       requires_grad=False)
            #              for i in range(config.n_agent)]
            #[thread,agent,obs]
            torch_obs = [torch.Tensor(ob).cuda() for ob in obs]
            # get actions as torch Variables
            #[thread,agent,act]
            actions = [model[i].step(torch_obs[i], explore=True)[0] for i in range(config.n_rollout_threads)]

            # rearrange actions to be per environment
            #[thread,agent,act]
            actions = [a.cpu().numpy() for a in actions]
            next_obs, rewards, dones, infos = env.step(actions)

            #分配各个option的经历

            for th in range(config.n_rollout_threads):
                
                _obs = np.expand_dims(obs[th],1)
                _actions = np.expand_dims(actions[th],0)
                _rewards = np.expand_dims(rewards[th],1)
                _times = np.full_like(_rewards,(et_i+1)/env.seconds_per_step)
                _next_obs = np.expand_dims(next_obs[th],1)
                _dones = np.expand_dims(dones[th],1)
                replay_buffer[th].push(_obs, _actions, _rewards, _next_obs, _dones,_times)

            
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer[0]) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                for i in range(config.n_rollout_threads):

                    if config.use_gpu:
                        model[i].prep_training(device='cuda')
                    else:
                        model[i].prep_training(device='cpu')
                    for u_i in range(config.num_updates):
                        sample = replay_buffer[i].sample(config.batch_size,
                                                    to_gpu=config.use_gpu)
                        model[i].optimize_model(sample)#model[0]
                        model[i].update_all_targets()
                    if config.use_gpu:
                        model[i].prep_rollouts(device='cuda')
                    else:
                        model[i].prep_rollouts(device='cpu')
                

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir , exist_ok=True)
            model[0].save(run_dir / f'model_ep{ep_i + 1}.pt')
        # ep_rews = replay_buffer.get_average_rewards(
        #    config.episode_length * config.n_rollout_threads)
        # for a_i, a_ep_rew in enumerate(ep_rews):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
        #                       a_ep_rew, ep_i)
        # all_ep_rews = sum(np.array(ep_rews))/config.n_rollout_threads
        # logger.add_scalar('episode_rewards_allagent',
        #                       ep_rew_mean, ep_i)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",default='Intersec', help="Name of environment")
    parser.add_argument("--env_name",default='jinan')
    parser.add_argument("--dqn", default=1, type=int)
    parser.add_argument("--n_agent", default=1, type=int)
    parser.add_argument("--test_interval", default=10, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e7), type=int)
    parser.add_argument("--n_episodes", default=200, type=int)
    parser.add_argument("--episode_length", default=3600, type=int)
    parser.add_argument("--steps_per_update", default=10, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=256, type=int,
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
    parser.add_argument("--load_model", default=True, type=bool)
    parser.add_argument("--model_path", default='models/Intersec/run4/model_ep61.pt')
    
    config = parser.parse_args()
    run(config, 0)
    # cProfile.run("")
