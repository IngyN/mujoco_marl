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
