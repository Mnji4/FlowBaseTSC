from env.myenv import *
if __name__ == '__main__':
    env = make_parallel_env('/home/zenianliang/Rainbow-TrafficLight/config/config_3.json',1,42)
    import time
    t0 = time.time()
    for i in range(3600):
        env.step_async(np.zeros((2,12)).astype(int))
        env.step_wait()
    print(time.time()-t0)