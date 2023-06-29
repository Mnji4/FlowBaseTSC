from os import times
import gym
import numpy as np
 # core.Env是gym的环境基类,自定义的环境就是根据自己的需要重写其中的方法；
 # 必须要重写的方法有:
 # __init__()：构造函数
 # reset()：初始化环境
 # step()：环境动作,即环境对agent的反馈
 # render()：如果要进行可视化则实现
import cityflow
import sys
import json
# roadnet
#f =  open("manhattan/hangzhou1/roadnet.json", 'r')
#f =  open("manhattan/4/roadnet_4.json", 'r')
#f =  open("manhattan/Jinan/3_4/roadnet_3_4.json", 'r')
#f =  open("manhattan/Hangzhou/4_4/roadnet_4_4.json", 'r')
#f =  open("manhattan/16_3/roadnet_16_3.json", 'r')
#a = json.load(f)
#f.close()
# a['intersections'][26]['trafficLight']['lightphases'][8] = a['intersections'][26]['trafficLight']['lightphases'][7]
# a['intersections'][26]['trafficLight']['lightphases'][7] = a['intersections'][26]['trafficLight']['lightphases'][6]
# a['intersections'][26]['trafficLight']['lightphases'][6]['availableRoadLinks'] = []
# json_save_path = 'manhattan/manhattan1.json'
# with open(json_save_path, 'w', encoding='utf-8') as file:
#     json.dump(a, file, ensure_ascii=False)

class MyEnv(gym.Env):
    def __init__(self, eng_config_file, max_step = 3600):
        f = open(eng_config_file, 'r')
        roadnet_file = f.readlines()[3][20:-4]
        f = open(roadnet_file, 'r')
        a = json.load(f)
        f.close()
        self.eng_config_file = eng_config_file
        self.eng = cityflow.Engine(self.eng_config_file, thread_num=1)
        self.eng.set_save_replay(True)
        self.reward_flag = 1
        self.pre_lane_vehicle = {}
        self.now_step = 0
        self.max_step = max_step
        self.action_space = gym.spaces.Discrete(9)  # 动作空间
        self.num_per_action = 30
        self.redtime = 3
        self.info_enable = 0
        self.intersections = {}
        self.agents = {}
        self.single_phase_agents = {}
        self.observation_features = ['lane_vehicle_num', 'waiting_vehicle_num']
        self.inverse_road = {}
        self.vehicle_start_time = {}
        for o in a['roads']:
            key1 = o['startIntersection'] + o['endIntersection']
            key2 = o['endIntersection'] + o['startIntersection']
            self.inverse_road[key1] = []
            self.inverse_road[key2] = []
        for o in a['roads']:
            key1 = o['startIntersection'] + o['endIntersection']
            key2 = o['endIntersection'] + o['startIntersection']
            self.inverse_road[key1].append(o)
            self.inverse_road[key2].append(o)
        self.tmp = {}

        for o in self.inverse_road.values():
            if len(o) == 2:
                self.tmp[o[0]['id']] = o[1]['id']
                self.tmp[o[1]['id']] = o[0]['id']
        self.inverse_road = self.tmp
        for o in a['intersections']:
            agent = 0
            if len(o['trafficLight']['lightphases']) == 9:
                #links = [l for o['trafficLight']['lightphases']]
                if len(set([d for p in o['trafficLight']['lightphases'] for d in p['availableRoadLinks']])) != max([len(p['availableRoadLinks']) for p in o['trafficLight']['lightphases']]):
                #     agent = 1
                #     road = [-1,-1,-1,-1,-1,-1,-1,-1]
                #     mask = [[[4,5],[2,5]],
                #             [[3,6],[1,6]],
                #             [[4,7],[2,7]],
                #             [[3,8],[1,8]]]

                #     for maski,maskr in enumerate(mask):
                #         ml = maskr[0]
                #         s1 = set(o['trafficLight']['lightphases'][ml[0]]['availableRoadLinks'])
                #         s2 = set(o['trafficLight']['lightphases'][ml[1]]['availableRoadLinks'])
                #         sr = set([i for i in range(len(o['roadLinks'])) if o['roadLinks'][i]['type'] == 'turn_right'])
                #         l = list(s1 & s2 - sr)
                #         if l:
                #             road[maski] = o['roadLinks'][l[0]]['startRoad']
                #             road[(maski+1)%4+4] = o['roadLinks'][l[0]]['endRoad']

                #         ms = maskr[1]
                #         s1 = set(o['trafficLight']['lightphases'][ms[0]]['availableRoadLinks'])
                #         s2 = set(o['trafficLight']['lightphases'][ms[1]]['availableRoadLinks'])
                #         sr = set([i for i in range(len(o['roadLinks'])) if o['roadLinks'][i]['type'] == 'turn_right'])
                #         l = list(s1 & s2 - sr)
                #         if l:
                #             road[maski] = o['roadLinks'][l[0]]['startRoad']
                #             road[(maski+2)%4+4] = o['roadLinks'][l[0]]['endRoad']

                #     direction = ['turn_left','go_straight','turn_right']
                    
                #     for l in o['roadLinks']:
                #         if l['endRoad'] in road:
                #             if l['endRoad'] not in self.inverse_road or l['startRoad'] != self.inverse_road[l['endRoad']]:
                #                 diff = direction.index(l['type'])
                #                 direct = road.index(l['endRoad'])
                #                 road[(3-diff+direct)%4] = l['startRoad']
                #     if len([o for o in road if o != -1]) != len(o['roads']):
                #         0 == 0
                #         pass
                    agent = 1
                    self.agents[o['id']] = o['roads']
                else:
                    self.single_phase_agents[o['id']] = o

            self.intersections[o['id']] = o
            self.intersections[o['id']]['have_signal'] = agent
        self.agentlist = list(self.agents.keys())
        self.roads = {}
        for o in a['roads']:

            self.roads[o['id']] = {
                'start_inter': o['startIntersection'],
                'end_inter': o['endIntersection'],
                'length': ((o['points'][0]['x']-o['points'][1]['x'])**2+(o['points'][0]['y']-o['points'][1]['y'])**2)**0.5,
                'speed_limit': float(o['lanes'][0]['maxSpeed']),
                'num_lanes': len(o['lanes']),
                # 'angle':o['_compass_angle']
            }

        # cloest = {}
        # mask = {}
        # for inter,roads in self.agents.items():
        #     ai = self.agentlist.index(inter)
        #     c = [ai,-1,-1,-1,-1]
        #     m = [1,1,1,1]
        #     for roadi,road in enumerate(roads[:4]):
        #         if self.roads[road]['start_inter'] in self.agentlist:
        #             c[roadi+1] = self.agentlist.index(self.roads[road]['start_inter'])
        #             m[roadi] = 0
        #     cloest[ai] = c
        #     mask[ai] = m
        # data = {}
        # data['cloest'] = cloest
        # data['mask'] = mask
        # np.save('data48.npy',data)
        agents = [o for o in a['intersections'] if o['id'] in self.agents]
        ob_space = {}
        # for agent in self.agents:
        #     for road in self.agents[agent]:
        #         if road != -1:
        #             for lane_i in range(self.roads[road]['num_lanes']):
        #                 ob_space["{}_{}".format(road,lane_i)] = gym.spaces.Box(low=-1e10, high=1e10, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-1e10, high=1e10, shape=(len(self.agents),24,))
		# 其他成员

        self.now_phases = {o:-1 for o in self.agents}
        for agent_id, roads in self.agents.items():
            for road in roads:
                for i in range(3):
                    lane = road + '_' + str(i)
                    self.pre_lane_vehicle[lane] = set()
        
        self.agent_lane = [[str(road) + '_' + str(i) for i in range(3) for road in agentroads[:4]] for agentroads in self.agents.values()]
        
    def _get_info(self):
        info = {}
        if(self.info_enable == 0):
            return info
        else:
            v_list = self.eng.get_vehicles()
            for vehicle in v_list:
                info[vehicle] = self.eng.get_vehicle_info(vehicle)
            return info

    def _get_reward(self):

        def get_diff(pre,sub):
            in_num = 0
            out_num = 0
            for vehicle in pre:
                if(vehicle not in sub):
                    out_num +=1
            for vehicle in sub:
                if(vehicle not in pre):
                    in_num += 1
            return in_num,out_num

        def get_pass(roads):
            pass_num = 0
            for road in roads:
                for i in range(3):
                    lane = road + '_' + str(i)
                    if(lane not in self.now_lane_vehicle.keys()):
                        self.now_lane_vehicle[lane] = set()
                    for vehicle in self.pre_lane_vehicle[lane]:
                        if vehicle not in self.now_lane_vehicle[lane] and vehicle in self.vehicles:
                            pass_num += 1
            return pass_num
        rwds = {}
        # return every
        self.now_lane_vehicle = self.eng.get_lane_vehicles()
        self.vehicles = self.eng.get_vehicles()
        for agent_id, roads in self.agents.items():
            rwds[agent_id] = get_pass(roads[:4])
        self.pre_lane_vehicle = self.now_lane_vehicle

        return np.array(list(rwds.values()))

    def _get_inter_tt(self):
        res = np.zeros(len(self.agentlist))
        for i in range(len(self.agentlist)):
            for lane in self.agent_lane[i]:
                res[i] += self.lane_step_travel_time[lane]
        return res
        
    # def _update_travel_time(self):
    #     now_vehicles = self.eng.get_vehicles()
    #     for v in now_vehicles:
    #         if v not in self.vehicle_start_time:
    #             vehicle_info = self.eng.get_vehicle_info(v)
    #             route_len = -float(vehicle_info['distance'])
    #             for road in vehicle_info['route'].strip().split(' '):
    #                 route_len += float(self.roads[road]['length'])
    #             self.vehicle_start_time[v] = [self.now_step, route_len/11.111, -1]
    #     for v in self.pre_vehicles:
    #         if v not in now_vehicles:
    #             self.vehicle_start_time[v][2] = (self.now_step - self.vehicle_start_time[v][0])/self.vehicle_start_time[v][1]
    #     self.pre_vehicles = now_vehicles

    def _update_travel_time(self):
        now_vehicles = self.eng.get_vehicles()
        for vehicle in now_vehicles:
            if vehicle not in self.vehicle_start_time:
                vehicle_info = self.eng.get_vehicle_info(vehicle)
                start_point = float(vehicle_info['distance'])
                self.vehicle_start_time[vehicle] = [self.now_step, start_point, 0, -1,vehicle_info['route'].strip().split(' ')]
            else:

                vehicle_info = self.eng.get_vehicle_info(vehicle)
                if 'road' not in vehicle_info:
                    now_road =vehicle_info['drivable'][vehicle_info['drivable'].find('TO_')+3:-2]
                    vehicle_info['distance'] = 0
                else:
                    now_road = vehicle_info['road']
                route = self.vehicle_start_time[vehicle][4]
                i = route.index(now_road)
                pass_len = -self.vehicle_start_time[vehicle][1] + float(vehicle_info['distance'])

                for road in route[:i]:
                    pass_len += float(self.roads[road]['length'])
                self.vehicle_start_time[vehicle][2] = pass_len/11.111
                #如果已走路程理想时间太短，说明出bug了，把它看作新进来的车
                if self.vehicle_start_time[vehicle][2]<1 or (self.now_step - self.vehicle_start_time[vehicle][0])/pass_len*11.111<0.9:
                    self.vehicle_start_time[vehicle][0] = self.now_step
                    self.vehicle_start_time[vehicle][1] = float(vehicle_info['distance'])
                    #已走路程理想时间
                    self.vehicle_start_time[vehicle][2] = 0
                    self.vehicle_start_time[vehicle][4] = vehicle_info['route'].strip().split(' ')

        for vehicle in self.pre_vehicles:
            if vehicle not in now_vehicles:
                route = self.vehicle_start_time[vehicle][4]
                pass_len = -self.vehicle_start_time[vehicle][1]
                for road in route:
                    pass_len += float(self.roads[road]['length'])
                self.vehicle_start_time[vehicle][3] = (self.now_step - self.vehicle_start_time[vehicle][0])/pass_len*11.111
                # if self.vehicle_start_time[vehicle][3] < 0.9:
                #     print()
        self.pre_vehicles = now_vehicles

    def _get_observations(self):
        # return self.eng.get_lane_vehicle_count()
        obs = {}
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        #vehicle_speed = self.eng.get_vehicle_speed()
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicle = self.eng.get_lane_vehicles()
        vehicle_speed = self.eng.get_vehicle_speed()
        self._update_travel_time()
        lane_delay = {}
        lane_travel_time_index = {}
        global_tti = 0.0
        vehicle_num = 0
        for lane,vehicles in lane_vehicle.items():
            delay = 0
            ttis = 0
            for vehicle in vehicles:
                delay += 11.111 - vehicle_speed[vehicle]
                if self.vehicle_start_time[vehicle][0] != self.now_step:
                    if self.vehicle_start_time[vehicle][2] == 0:
                        travel_time_index = (self.now_step-self.vehicle_start_time[vehicle][0])/5
                    else:
                        travel_time_index = (self.now_step-self.vehicle_start_time[vehicle][0])/(self.vehicle_start_time[vehicle][2])

                    if travel_time_index>30:
                        travel_time_index = 30
                    # if travel_time_index<1:
                    #     travel_time_index = 1
                    if travel_time_index < 0.9:
                        print()
                    global_tti += travel_time_index
                    ttis += travel_time_index
                    vehicle_num += 1
            lane_delay[lane] = delay
            lane_travel_time_index[lane] = ttis
        delay_sum = sum(list(lane_delay.values()))

        extra_tti = {}
        if vehicle_num != 0:
            l = np.array(list(self.vehicle_start_time.values()))[:,3]
            l = l[l != -1]
            mean_tti = (global_tti + l.sum())/(vehicle_num + len(l))
        for lane,vehicles in lane_vehicle.items():
            etti = 0
            for vehicle in vehicles:
                if self.vehicle_start_time[vehicle][0] != self.now_step:
                    if self.vehicle_start_time[vehicle][2] == 0:
                        travel_time_index = (self.now_step-self.vehicle_start_time[vehicle][0])/5
                    else:
                        travel_time_index = (self.now_step-self.vehicle_start_time[vehicle][0])/(self.vehicle_start_time[vehicle][2])
                    if travel_time_index>30:
                        travel_time_index = 30
                    # if travel_time_index<1:
                    #     travel_time_index = 1
                    if travel_time_index < 0.9:
                        print()
                    if travel_time_index> mean_tti:
                    #if 1:
                        etti += abs((travel_time_index - mean_tti)/mean_tti)**1.1
            extra_tti[lane] = etti
        


        # add 1 dimension to give current step for fixed time agent
        obs = np.full((len(self.agents),64),-1)

                
        direction = ['turn_left','go_straight','turn_right']
        for ai,agent in enumerate(self.agents):
            roads = self.agents[agent]
            inter = self.intersections[agent]
            for roadlink in inter['roadLinks']:
                lanes = list(set([l['startLaneIndex'] for l in roadlink['laneLinks']]))
                movement = roads.index(roadlink['startRoad'])*3 + direction.index(roadlink['type'])
                if obs[ai][movement] == -1:
                    obs[ai][movement] = 0
                    obs[ai][movement+12] = 0
                    obs[ai][movement+24] = 0
                    obs[ai][movement+36] = 0
                for lanei in lanes:
                    lane = roadlink['startRoad']+'_'+str(lanei)
                    obs[ai][movement] += lane_vehicle_count[lane]
                    obs[ai][movement+12] += lane_waiting_vehicle_count[lane]
                    obs[ai][movement+24] += lane_delay[lane]
                    obs[ai][movement+36] += extra_tti[lane]
                    
                    # if obs[ai][movement]:
                    #     obs[ai][movement+24] = (lane_waiting_vehicle_count[lane] * lane_vehicle_count[lane] + (obs[ai][movement] - lane_vehicle_count[lane]) * obs[ai][movement+24]) / obs[ai][movement]

                    
            for i,endroad in enumerate(roads[4:]):
                if obs[ai][i+48] == -1:
                    obs[ai][i+48] = 0
                    obs[ai][i+52] = 0
                    obs[ai][i+56] = 0
                    obs[ai][i+60] = 0
                for lanei in range(self.roads[endroad]['num_lanes']):
                    lane = endroad+'_'+str(lanei)
                    obs[ai][i+48] += lane_vehicle_count[lane]
                    obs[ai][i+52] += lane_waiting_vehicle_count[lane]
                    obs[ai][i+56] += lane_delay[lane]
                    obs[ai][i+60] += extra_tti[lane]
                    # if obs[ai][i+36]:
                    #     obs[ai][i+44] = (lane_waiting_vehicle_count[lane] * lane_vehicle_count[lane] + (obs[ai][i+36] - lane_vehicle_count[lane]) * obs[ai][i+44]) / obs[ai][i+36]
            #obs[ai][48] += delay_sum
        
        
        
        return obs

    def _get_dones(self):
        #
        dones = {}
        for agent_id in self.agents.keys():
            dones[agent_id] = self.now_step >= self.max_step

        return dones

    def run_whole_phase(self,action):
        self.lane_step_travel_time = {k:0 for i in range(len(self.agent_lane)) for k in self.agent_lane[i]}
        
        for agent_id,phase in action.items():
            if action[agent_id] == self.now_phases[agent_id] or self.now_phases[agent_id] == -1:
                self.eng.set_tl_phase(agent_id,phase)
                
            else:
                self.eng.set_tl_phase(agent_id,0)
        #决策间隔小于红灯时间
        if self.num_per_action <=self.redtime:
            for t in range(self.num_per_action):
                lane_v_cnt = self.eng.get_lane_vehicle_count()
                for lane in self.lane_step_travel_time.keys():
                    self.lane_step_travel_time[lane] += lane_v_cnt[lane]
                self.eng.next_step()
        #决策间隔大于红灯时间
        else:
            for t in range(self.redtime):
                lane_v_cnt = self.eng.get_lane_vehicle_count()
                for lane in self.lane_step_travel_time.keys():
                    self.lane_step_travel_time[lane] += lane_v_cnt[lane]
                self.eng.next_step()

            for agent_id,phase in action.items():
                self.eng.set_tl_phase(agent_id,phase)

            for t in range(self.redtime,self.num_per_action):
                lane_v_cnt = self.eng.get_lane_vehicle_count()
                for lane in self.lane_step_travel_time.keys():
                    self.lane_step_travel_time[lane] += lane_v_cnt[lane]
                self.eng.next_step()
                if t == 29:
                    if self.reward_flag:
                        self.reward += self._get_reward()
                    
        self.now_step+=self.num_per_action
        self.now_phases = action
    def step(self, action):
        # here action is a dict {agent_id:phase}
        self.reward = np.zeros(len(self.agentlist))
        self.run_whole_phase(action)


        reward = self._get_inter_tt()#_get_reward
        dones = self._get_dones()
        obs = self._get_observations()
        info = {}#self._get_reward()self._get_info()

        if self.now_step == 3600:
            for v,l in self.vehicle_start_time.items():
                if l[3] == -1:
                    #计算已走路程的理想时间
                    vehicle_info = self.eng.get_vehicle_info(v)
                    if 'road' not in vehicle_info:
                        now_road =vehicle_info['drivable'][vehicle_info['drivable'].find('TO_')+3:-2]
                        vehicle_info['distance'] = 0
                    else:
                        now_road = vehicle_info['road']

                    route = l[4]
                    i = route.index(now_road)
                    pass_len = -l[1] + float(vehicle_info['distance'])

                    for road in route[:i]:
                        pass_len += float(self.roads[road]['length'])
                    if pass_len == 0:
                        l[3] = 30
                    else:
                        l[3] = min((3600 - l[0]) / pass_len*11.111,30)
                    # if l[3] < 0.9:
                    #     print()
                    #
                    # route = vehicle_info['route'].strip().split(' ')
                    # i = route.index(now_road)
                    # route_to_drive = []
                    # len_to_drive = float(self.roads[now_road]['length']) - float(vehicle_info['distance'])
                    # if i != len(route)-1:
                    #     route_to_drive = route[i+1:]
                    # for road in route_to_drive:
                    #     len_to_drive += float(self.roads[road]['length'])
                    # ideal_time = len_to_drive/11.111
                    # l[3] = (3600 - l[0] + ideal_time) /ideal_time
                
            atti = np.array(list(self.vehicle_start_time.values()))[:,3].mean()
            vtti = (np.array(list(self.vehicle_start_time.values()))[:,3]/atti).var()
            print('average time:{}, average tti:{}, tti variance:{}'.format(self.eng.get_average_travel_time(),atti,vtti))
            info = (self.eng.get_average_travel_time(),atti,vtti)
            # print('average time:{}'.format(self.eng.get_average_travel_time()))
        return obs, reward, dones , info

    def reset(self):
        self.eng = cityflow.Engine(self.eng_config_file, thread_num = 1)
        self.pre_lane_vehicle = {o:set() for o in self.pre_lane_vehicle}
        self.pre_vehicles = {}
        #self.eng.reset()
        self.now_step = 0
        self.now_phases = {o:-1 for o in self.agents}
        self.vehicle_start_time = {}
        return self._get_observations(),self._get_info()
# env = MyEnv('manhattan/config_48.json')

# pass
# action = {a:0 for a in env.agents}
# env.step(action)
# for i in range(3570):
#     env.eng.next_step()
# print('average time:{}'.format(env.eng.get_average_travel_time()))


