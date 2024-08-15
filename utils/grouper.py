import numpy as np
class Grouper():
    def __init__(self, env):
        super().__init__()
        self.n = len(env.agents)
        self.in_group = np.zeros(self.n)
        self.group_pos = np.full((self.n,2),-1)
        self.groups = np.full((self.n//2,2,2),-1)
        self.xy = env.xy
        self.num_x = len(set(self.xy[:,0]))
        self.num_y = len(set(self.xy[:,1]))
    def grouping_ew(self):
        return       
     
    def grouping_ns(self):
        groupi = 0
        for j in range(self.num_y):
            for i in range(0,self.num_x,2):
                # if(j==0 and i<2):continue
                print(self.xy[i*self.num_y+j],self.xy[(i+1)*self.num_y+j])
                self.in_group[i*self.num_y+j]=1
                self.in_group[(i+1)*self.num_y+j]=1
                self.groups[groupi][0] = self.xy[i*self.num_y+j]
                self.groups[groupi][1] = self.xy[(i+1)*self.num_y+j]
                self.group_pos[i*self.num_y+j] = np.array([groupi,0])
                self.group_pos[(i+1)*self.num_y+j] = np.array([groupi,1])
                groupi+=1
        return 

    def to_group_obs(self, obs):
        # obs = np.arange(36).reshape(1,12,3)
        batch_size, n, obs_dim = obs.shape
        
        # 计算有分组和无分组的智能体数量
        grouped_agents = np.sum(self.in_group == 1)
        ungrouped_agents = np.sum(self.in_group == 0)
        
        # 创建结果数组
        paired_obs = []
        unpaired_obs = []
        
        # 处理有分组的智能体
        if grouped_agents > 0:
            agent1_indices = np.where((self.group_pos[:, 0] != -1) & (self.group_pos[:, 1] == 0))[0]
            agent2_indices = np.where((self.group_pos[:, 0] != -1) & (self.group_pos[:, 1] == 1))[0]
            paired_obs = np.zeros((2, batch_size, grouped_agents // 2, obs_dim))
            paired_obs[0, :, :, :] = obs[:, agent1_indices, :]
            paired_obs[1, :, :, :] = obs[:, agent2_indices, :]
        
        # 处理未分组的智能体
        if ungrouped_agents > 0:
            unpaired_indices = np.where(self.in_group == 0)[0]
            unpaired_obs = obs[:, unpaired_indices, :]
        return paired_obs, unpaired_obs
    
    def parse_group_action(self, paired_action, unpaired_action):
        batch_size = paired_action.shape[1] if paired_action is not None else unpaired_action.shape[0]
        action_dim = paired_action.shape[-1] if paired_action is not None else unpaired_action.shape[-1]
        
        # Initialize the full action array
        full_action = np.zeros((batch_size, self.n, action_dim))
        
        # Fill in paired actions
        if paired_action is not None:
            agent1_indices = np.where((self.group_pos[:, 0] != -1) & (self.group_pos[:, 1] == 0))[0]
            agent2_indices = np.where((self.group_pos[:, 0] != -1) & (self.group_pos[:, 1] == 1))[0]
            full_action[:, agent1_indices, :] = paired_action[0]
            full_action[:, agent2_indices, :] = paired_action[1]
        
        # Fill in unpaired actions
        if unpaired_action is not None:
            unpaired_indices = np.where(self.in_group == 0)[0]
            full_action[:, unpaired_indices, :] = unpaired_action
        
        return full_action