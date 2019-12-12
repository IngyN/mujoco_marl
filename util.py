import numpy as np
from collections import OrderedDict

# Converts an env.action_space.sample() to the proper format
def convert_action(a):
    action = OrderedDict([])

    for k in a[0]:
        temp = []
        for agent in a:
            temp.append(agent[k])
        action[k] = np.array(temp)

    return action

def random_action():
    output = OrderedDict([])
    output['action_movement'] = np.random.randint(0,10+1,3)
    output['action_pull'] = np.random.randint(0,1+1)
    output['action_glueall'] = np.random.randint(0,1+1)
    return output
def no_action():
    output = OrderedDict([])
    output['action_movement'] = np.array([5,5,5])
    output['action_pull'] = np.array(0)
    output['action_glueall'] = np.array(0)
    return output
