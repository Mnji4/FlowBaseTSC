from env.myenv import *
if __name__ == '__main__':
    env = make_parallel_env('/home/zenianliang/FlowBaseTSC/config/config_jinan.json',2,42)
    import time
    t0 = time.time()
    for i in range(360):
        env.step_async(np.random.randint(1, 8, size=(2, 12)))
        env.step_wait()
    print(time.time()-t0)