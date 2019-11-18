import numpy as np
from collections import OrderedDict

# Converts an env.action_space.sample() to the proper format
def convert_action(a):
    action = OrderedDict([])

    for k in a:
        action[k] = np.array(a[k])

    return action