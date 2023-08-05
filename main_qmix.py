import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBufferTime
# from algorithms.attention_sac1 import AttentionSAC
#from algorithms.attention_ppo1 import AttentionPPO
from algorithms.distral import Distral
import gym
import json
from utils.ma_env_time import MaEnv,make_env,make_parallel_env


def run(config, start = 0):
    #count_cloest()
    best_rew = 0

    
    model_dir = Path('manhattan/models') / config.env_id / config.model_name
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
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
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
                                [config.a_dim for i in range(1)])
        replay_buffer.append(replay_buffer_)
    #filename = Path('other/MAACcbe/models/CBE/colight_sac_round2_pressure_r/run2/incremental/model_ep11.pt')                                  
    #model = AttentionSAC.init_from_save(filename, load_critic=True)
    meta_model = Distral.init_from_env(env,  s_dim=config.s_dim,
                                        a_dim= config.a_dim,
                                        n_agent= 1,
                                        tau=config.tau,
                                        pi_lr=config.pi_lr,
                                        q_lr=config.pi_lr,
                                        gamma=config.gamma,
                                        pol_hidden_dim=config.pol_hidden_dim,
                                        critic_hidden_dim=config.critic_hidden_dim,
                                        attend_heads=config.attend_heads,
                                        reward_scale=config.reward_scale)
    if config.load_model:
        for i in range(2):
            filename = Path(config.model_path + 'model%i_ep121.pt' %(i))
            model_ = Distral.init_from_save(filename, load_critic=True)
            model.append(model_)
        filename = Path(config.meta_model_path + 'meta_model_ep121.pt')
        meta_model = Distral.init_from_save(filename, load_critic=True)


    t = 0
    
    #inter_region = data['']
    train_to_log = []
    test_central_to_log = []
    test_policies_to_log = []
    batchsizes = [0,0]
    for ep_i in range(start, config.n_episodes, config.n_rollout_threads):
        if ep_i % config.test_interval < config.n_rollout_threads:
            print('testing central policy')
            obs = env.reset()
            for test_t in range(0,3600,config.interval_length):
                torch_obs = [torch.Tensor(ob) for ob in obs]
                # get actions as torch Variables
                #[thread,agent,act]
                if config.dqn:
                    actions = [model[0].step([torch_obs[i]], explore=False)[0] for i in range(config.n_rollout_threads)]
                else:
                    actions = [meta_model.step([torch_obs[i]], explore=False)[0] for i in range(config.n_rollout_threads)]
                # rearrange actions to be per environment
                #[thread,agent,act]
                actions = [a.numpy() for a in actions]
                next_obs, rewards, dones, infos = env.step(actions)
                obs = next_obs
            infos = np.array([infos[i][1] for i in range(config.n_rollout_threads)])
            test_central_to_log.append(infos)
            logger.add_scalar('testing/central att',infos.mean(0)[0], ep_i)
            logger.add_scalar('testing/central att index',infos.mean(0)[1], ep_i)
            logger.add_scalar('testing/central att index var',infos.mean(0)[2], ep_i)


            print('testing policies')
            obs = env.reset()
            for test_t in range(0,3600,config.interval_length):
                torch_obs = [torch.Tensor(ob) for ob in obs]
                # get actions as torch Variables
                #[thread,agent,act]
                actions = [model[i].step([torch_obs[i]], explore=False)[0] for i in range(config.n_rollout_threads)]

                # rearrange actions to be per environment
                #[thread,agent,act]
                actions = [a.numpy() for a in actions]
                next_obs, rewards, dones, infos = env.step(actions)
                obs = next_obs
            infos = np.array([infos[i][1] for i in range(config.n_rollout_threads)])
            s = infos[0][0]+infos[1][0]
            p1 = (infos[0][0]/s)**2
            p2 = (infos[1][0]/s)**2
            batchsizes[1] = int(p1/(p1+p2)*256)
            batchsizes[0] = 256 - batchsizes[1]
            test_policies_to_log.append(infos)
            for i in range(config.n_rollout_threads):
                logger.add_scalar('testing/model%i att' %i,infos[i][0], ep_i)
                logger.add_scalar('testing/model%i att index' %i,infos[i][1], ep_i)
                logger.add_scalar('testing/model%i att index var' %i,infos[i][2], ep_i)


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
            actions = [model[i].step([torch_obs[i]], explore=True)[0] for i in range(config.n_rollout_threads)]

            # rearrange actions to be per environment
            #[thread,agent,act]
            actions = [a.numpy() for a in actions]
            next_obs, rewards, dones, infos = env.step(actions)
            r_step = np.array([sum(o) for o in rewards])
            ep_rew_mean = (ep_rew_mean * et_i + r_step*config.interval_length)/(et_i+config.interval_length)


            if et_i%30000 == 0:
                #print([(np.argmax(actions[0][i])+1) for i in range(22)])
                print('ep %i step %i' %(ep_i,et_i))
                print(r_step)
                print(ep_rew_mean)
            #分配各个option的经历

            for th in range(config.n_rollout_threads):
                
                _obs = np.expand_dims(obs[th],1)
                _actions = np.expand_dims(actions[th],0)
                _rewards = np.expand_dims(infos[th][0][th],1)
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
                        model[i].optimize_model(sample, meta_model.critic)#model[0]
                        model[i].update_all_targets()
                    model[i].prep_rollouts(device='cpu')
                
                if config.use_gpu:
                    meta_model.prep_training(device='gpu')
                else:
                    meta_model.prep_training(device='cpu')

                for meta_u_i in range(4):
                    samples = [replay_buffer[i].sample(batchsizes[i],
                                                        to_gpu=config.use_gpu) for i in range(config.n_rollout_threads)]
                    if config.dqn == 0:
                        meta_model.optimize_policy(samples,model)
                meta_model.prep_rollouts(device='cpu')
        infos = np.array([infos[i][1] for i in range(config.n_rollout_threads)])
        train_to_log.append(infos)
        for i in range(config.n_rollout_threads):
            logger.add_scalar('training/model%i att' %i,infos[i][0], ep_i)
            logger.add_scalar('training/model%i att index' %i,infos[i][1], ep_i)
            logger.add_scalar('training/model%i att index var' %i,infos[i][2], ep_i)


        # ep_rews = replay_buffer.get_average_rewards(
        #    config.episode_length * config.n_rollout_threads)
        # for a_i, a_ep_rew in enumerate(ep_rews):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
        #                       a_ep_rew, ep_i)
        # all_ep_rews = sum(np.array(ep_rews))/config.n_rollout_threads
        # logger.add_scalar('episode_rewards_allagent',
        #                       ep_rew_mean, ep_i)
            

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            for i in range(2):
                model[i].save(run_dir / ('model%i_ep%i.pt' % (i,ep_i + 1)))
                meta_model.save(run_dir / ('meta_model_ep%i.pt' % (ep_i + 1)))


        #     meta_model.prep_rollouts(device='cpu')
        #     meta_model.save(run_dir / 'incremental' / ('meta_model_ep%i.pt' % (ep_i)))
        #     meta_model.save(run_dir / 'meta_model.pt')
    # train_to_log = []
    # test_central_to_log = []
    # test_policies_to_log = []
    dict = {
        'train_to_log':train_to_log,
        'test_central_to_log':test_central_to_log,
        'test_policies_to_log':test_policies_to_log
    }
    np.save(log_dir/'dict.npy',dict)
    r = np.array(test_policies_to_log)
    for th in range(config.n_rollout_threads):
        att = r[:,th,1]
        var = r[:,th,2]
        l = np.array((att,var))
        l = l[:,l.argsort(1)[0]]
        for i in range(l.shape[1]):
            logger.add_scalar('testing/model%i atti-var' %th,l[1][i], l[0][i])

    r = np.array(test_central_to_log)
    att = r.mean(1)[:,1]
    var = r.mean(1)[:,2]
    l = np.array((att,var))
    l = l[:,l.argsort(1)[0]]
    for i in range(l.shape[1]):
        logger.add_scalar('testing/central atti-var',l[1][i], l[0][i])

    r = np.array(train_to_log)
    for th in range(config.n_rollout_threads):
        att = r[:,th,1]
        var = r[:,th,2]
        l = np.array((att,var))
        l = l[:,l.argsort(1)[0]]
        for i in range(l.shape[1]):
            logger.add_scalar('training/model%i atti-var' %th,l[1][i], l[0][i])


    for i in range(2):
        model[i].save(run_dir / ('model%i_ep%i.pt' % (i,ep_i + 1)))
        meta_model.save(run_dir / ('meta_model_ep%i.pt' % (ep_i + 1)))
        env.close()
        logger.export_scalars_to_json(str(log_dir / 'summary.json'))
        logger.close()


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
    parser.add_argument("--n_agent", default=1, type=int)
    parser.add_argument("--interval_length", default=30, type=int)
    parser.add_argument("--test_interval", default=10, type=int)
    parser.add_argument("--n_rollout_threads", default=2, type=int)
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
    parser.add_argument("--meta_model_path", default='/start/manhattan/models/Distral/test/run109/')
    config = parser.parse_args()

    run(config, 0)
