from env.myenv import *
import cProfile
from utils.buffer import ReplayBufferTime
def run():
    env = make_parallel_env('config/config_3.json',1,42)
    replay_buffer = ReplayBufferTime(3600, 1,
                                [env.observation_space.shape[1] for i in range(1)],
                                [env.action_space.n for i in range(1)])
    env.env.traj_buffer = replay_buffer
    import time
    t0 = time.time()
    for i in range(3600):
        env.step_async(np.random.randint(0, 7, size=(1,len(env.env.agentlist))))
        env.step_wait()
    print(time.time()-t0)

if __name__ == '__main__':
    run()
    # cProfile.run("run()",  sort="cumulative")