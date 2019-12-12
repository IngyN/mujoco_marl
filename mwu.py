import numpy as np
from collections import OrderedDict

class MWU():
    def __init__(self):
        self.weights = [[1/11]*11,[1/11]*11,[1/11]*11,[1/2]*2,[1/2]*2]
        self.epsilon = 1e-3
    def set_weights(self,new_weights):
        self.weights = new_weights
    def sample(self):
        temp = [0]*len(self.weights)
        for i,row in enumerate(self.weights):
            r = np.random.rand()
            idx = 0
            while r > row[idx]:
                r-=row[idx]
                idx+=1
            temp[i] = idx
        output = OrderedDict([])
        output['action_movement'] = np.array(temp[0:3])
        output['action_pull'] = np.array(temp[3])
        output['action_glueall'] = np.array(temp[4])
        return output
    def backward(self,action,rew):
        temp = [0]*5
        temp[0] = action['action_movement'][0]
        temp[1] = action['action_movement'][1]
        temp[2] = action['action_movement'][2]
        temp[3] = action['action_pull']
        temp[4] = action['action_glueall']
        cost = (1-rew)/2 # Translate from [-1,1] to [0,1]
        for i,a in enumerate(temp):
            self.weights[i][a] = self.weights[i][a]*(1-self.epsilon*cost)
            norm = sum(self.weights[i])
            for j in range(len(self.weights[i])):
                self.weights[i][j] = self.weights[i][j]/norm

