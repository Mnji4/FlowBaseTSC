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
                print(self.xy[i*self.num_y+j],self.xy[(i+1)*self.num_y+j])
                self.in_group[i*self.num_y+j]=1
                self.in_group[(i+1)*self.num_y+j]=1
                self.groups[groupi][0] = self.xy[i*self.num_y+j]
                self.groups[groupi][1] = self.xy[(i+1)*self.num_y+j]
                self.group_pos[i*self.num_y+j] = np.array([groupi,0])
                self.group_pos[(i+1)*self.num_y+j] = np.array([groupi,1])
                groupi+=1
        return 