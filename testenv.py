from env.myenv import *
import cProfile

def run():
    env = make_parallel_env('config/config_3.json',1,42)
    import time
    t0 = time.time()
    for i in range(360):
        env.step_async(np.random.randint(0, 7, size=(1,len(env.env.agentlist))))
        env.step_wait()
    print(time.time()-t0)

if __name__ == '__main__':
    run()
    # cProfile.run("run()",  sort="cumulative")