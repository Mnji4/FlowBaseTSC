from env.myenv import *
import cProfile

def run():
    env = MyEnv('config/config_jinan.json',7200)
    for i in range(7200//env.seconds_per_step):
        env.step(np.random.randint(0, 7, size=(len(env.agentlist))))
        if env.now_step == 7200:
            for k,v in env.semi_buffer.items():
                if(len(v)):
                    print(k,v)
            print(f"pass {env.passnum} catch {env.catchnum}")
            print(sum(len(d) for d in env.vehicle_duration.values()))
            for lane,d in env.vehicle_duration.items():
                for k,v in d.items():
                    print(lane,k,v,v[1]-v[0])
            # print({k:len(v) for k,v in env.semi_buffer.items()})
    quit()
    env = make_parallel_env('config/config_jinan.json',2,42)
    import time
    t0 = time.time()
    for i in range(360):
        env.step_async(np.random.randint(0, 7, size=(len(env.agentlist))))
        env.step_wait()
    print(time.time()-t0)

if __name__ == '__main__':
    # run()
    cProfile.run("run()",  sort="cumulative")