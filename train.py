"""
This file includes the model and environment setup and the main training loop.
Look at the README.md file for details on how to use this.
"""

import time, random
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn

import torch, wandb
import numpy as np
from tqdm import trange
from rich import print

from common import argp
from common.rainbow import Rainbow
from env.myenv import make_parallel_env
from common.utils import LinearSchedule
from glob import glob
torch.backends.cudnn.benchmark = True  # let cudnn heuristics choose fastest conv algorithm

if __name__ == '__main__':
    args, wandb_log_config = argp.read_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set up logging & model checkpoints
    wandb.init(project='rainbow', save_code=True, config=dict(**wandb_log_config, log_version=100),
               mode=('online' if args.use_wandb else 'offline'), anonymous='allow', tags=[args.wandb_tag] if args.wandb_tag else [])
    runs = glob('checkpoints/*')
    runinx = 0
    if len(runs):
        runinx = max([int(x.split('/')[-1]) for x in runs])+1
    save_dir = Path("checkpoints") / f'{runinx}'
    save_dir.mkdir(parents=True)
    args.save_dir = str(save_dir)

    # create decay schedules for dqn's exploration epsilon and per's importance sampling (beta) parameter
    eps_schedule = LinearSchedule(0, initial_value=args.init_eps, final_value=args.final_eps, decay_time=args.eps_decay_frames)
    per_beta_schedule = LinearSchedule(0, initial_value=args.prioritized_er_beta0, final_value=1.0, decay_time=args.prioritized_er_time)

    # When using many (e.g. 64) environments in parallel, having all of them be correlated can be an issue.
    # To avoid this, we estimate the mean episode length for this environment and then take i*(mean ep length/parallel envs count)
    # random steps in the i'th environment.
    decorr_steps = None
    env = make_parallel_env('./config/config_jinan.json',2,36)
    num_intersection = len(env.agent_types)
    num_envs = env.num_envs
    args.parallel_envs = num_envs*num_intersection
    print(f'Creating', num_envs, 'and decorrelating environment instances. This may take up to a few minutes.. ', end='')
    states = env.reset()
    states = states.reshape(num_intersection * num_envs,-1)

    rainbow = Rainbow(env, args)
    wandb.watch(rainbow.q_policy)

    print('[blue bold]Running environment =', args.env_name,
          '[blue bold]\nwith action space   =', env.action_space,
          '[blue bold]\nobservation space   =', env.observation_space,
          '[blue bold]\nand config:', sn(**wandb_log_config))

    episode_count = 0
    returns = deque(maxlen=100)
    discounted_returns = deque(maxlen=10)
    losses = deque(maxlen=10)
    q_values = deque(maxlen=10)
    grad_norms = deque(maxlen=10)
    iter_times = deque(maxlen=10)

    returns_all = []
    q_values_all = []

    # main training loop:
    # we will do a total of args.training_frames/args.parallel_envs iterations
    # in each iteration we perform one interaction step in each of the args.parallel_envs environments,
    # and args.train_count training steps on batches of size args.batch_size
    t = trange(0, args.training_frames + 1, args.parallel_envs)
    for game_frame in t:
        iter_start = time.time()
        eps = eps_schedule(game_frame)
        per_beta = per_beta_schedule(game_frame)

        # reset the noisy-nets noise in the policy
        # if args.noisy_dqn:
            # rainbow.reset_noise(rainbow.q_policy)

        # compute actions to take in all parallel envs, asynchronously start environment step
        if(game_frame//(3600/env.seconds_per_step))%5<4:
            actions = rainbow.act(states, explore = True)
        else:
            actions = rainbow.act(states, explore = False)
        actions = actions.reshape((num_envs,num_intersection))
        env.step_async(actions)

        # if training has started, perform args.train_count training steps, each on a batch of size args.batch_size
        if rainbow.buffer.burnedin and game_frame%120 == 0:
            for train_iter in range(args.train_count):
                if args.noisy_dqn and train_iter > 0: rainbow.reset_noise(rainbow.q_policy)
                q, loss, grad_norm = rainbow.train(args.batch_size, beta=per_beta)
                losses.append(loss)
                grad_norms.append(grad_norm)
                q_values.append(q)
                q_values_all.append((game_frame, q))

        # copy the Q-policy weights over to the Q-target net
        # (see also https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/launcher.py#L155)
        if game_frame % args.sync_dqn_target_every == 0 and rainbow.buffer.burnedin:
            rainbow.sync_Q_target()

        # block until environments are ready, then collect transitions and add them to the replay buffer
        next_states, rewards, dones, infos = env.step_wait()
        next_states = next_states.reshape(num_envs*num_intersection,-1)
        actions = actions.reshape(num_envs*num_intersection,-1)
        rewards = rewards.reshape(num_envs*num_intersection)
        dones = dones.reshape(num_envs*num_intersection)
        for state, action, reward, done, j in zip(states, actions, rewards, dones, range(args.parallel_envs)):
            rainbow.buffer.put(state, action, reward, done, j=j)
        states = next_states

        # if any of the envs finished an episode, log stats to wandb
        for info, j in zip(infos, range(args.parallel_envs)):
            if t.n % 3600 == 0:
                # episode_metrics = info['episode_metrics']
                # returns.append(episode_metrics['return'])
                # returns_all.append((game_frame, episode_metrics['return']))
                # discounted_returns.append(episode_metrics['discounted_return'])

                log = {'x/game_frame': game_frame + j, 'x/episode': episode_count, 'grad_norm': np.mean(grad_norms),
                       'mean_loss': np.mean(losses), 'mean_q_value': np.mean(q_values), 'fps': args.parallel_envs / np.mean(iter_times),
                       'lr': rainbow.opt.param_groups[0]['lr']}
                if args.prioritized_er: log['per_beta'] = per_beta
                if eps > 0: log['epsilon'] = eps

                # log video recordings if available
                wandb.log(log)
                episode_count += 1

        if game_frame % (50_000-(50_000 % args.parallel_envs)) == 0:
            # print(f' [{game_frame:>8} frames, {episode_count:>5} episodes] running average return = {np.mean(returns)}')
            torch.cuda.empty_cache()

        # every 1M frames, save a model checkpoint to disk and wandb
        if game_frame % (500_000-(500_000 % args.parallel_envs)) == 0 and game_frame > 0:
            rainbow.save(game_frame, args=args, run_name=wandb.run.name, run_id=wandb.run.id, target_metric=np.mean(returns), returns_all=returns_all, q_values_all=q_values_all)
            print(f'Model saved at {game_frame} frames.')

        iter_times.append(time.time() - iter_start)
        t.set_description(f' [{game_frame:>8} frames, {episode_count:>5} episodes]', refresh=False)

    wandb.log({'x/game_frame': game_frame + args.parallel_envs, 'x/episode': episode_count,
               'x/train_step': (game_frame + args.parallel_envs) // args.parallel_envs * args.train_count,
               'x/emulator_frame': (game_frame + args.parallel_envs) * args.frame_skip})
    env.close()
    wandb.finish()
