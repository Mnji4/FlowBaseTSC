from env.myenv import *
import cProfile

def run():
    env = MyEnv('config/config_jinan.json',7200)
    for i in range(7200//env.seconds_per_step):
        env.step(np.random.randint(0, 7, size=(len(env.agentlist))))
    quit()
    env = make_parallel_env('config/config_jinan.json',2,42)
    import time
    t0 = time.time()
    for i in range(360):
        env.step_async(np.random.randint(1, 8, size=(2, 12)))
        env.step_wait()
    print(time.time()-t0)

if __name__ == '__main__':
    run()
    # cProfile.run("run()",  sort="cumulative")