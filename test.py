from mae_envs.envs import hide_and_seek
from util import convert_action
from agent import get_agent
from copy import deepcopy
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# environment parameters (there's more). Check make_env
hiders = 1
seekers = 1
boxes = 1
ramps = 1
food = 0
rooms = 2

display = True
load_weights = False

env = hide_and_seek.make_env(n_hiders=hiders, n_seekers=seekers, n_boxes=boxes, n_ramps=ramps, n_food=food, n_rooms=rooms)

# # probably shouldn't use those two. but was testing.
# rewardWrapper = hide_and_seek.HideAndSeekRewardWrapper(env, n_hiders=hiders, n_seekers=seekers)
# trackStatW = hide_and_seek.TrackStatWrapper(env, boxes, ramps, food)

# run one episode
env.seed(42)
env.reset()

agents = []
for i in range(hiders+seekers):
    agents.append(get_agent(env,i))
    if load_weights:
        agents[-1].load_weights("agent_%i_weights.h5f"%(i))

#https://github.com/keras-rl/keras-rl/blob/master/rl/core.py
obs = None
rew = None
done = None
info = None
agent_obs = [[]]*(hiders+seekers)
agent_act = [[]]*(hiders+seekers)
if display:
    env.render()
for a in agents:
    a.training = True

for e in range(1000):
    for t in range(5000):
        if obs is None:
            obs = deepcopy(env.reset())
            rew = np.float32(0)
            for i in range(hiders+seekers):
                agent_obs[i] = agents[i].processor.process_observation(obs)
        for i in range(hiders+seekers):
            action_i = agents[i].forward(agent_obs[i])
            agent_act[i] = agents[i].processor.process_action(action_i)
            done = False
        action = convert_action(agent_act)
        obs, rew, done, info = env.step(action)
        if display:
            env.render()
        obs = deepcopy(obs)
        if done:
            print("done")
            for i in range(hiders+seekers):
                agents[i].forward(agent_obs[i])
                agents[i].backward(0., terminal=False)
            obs = None
            break
        for i in range(hiders+seekers):
            metrics = agents[i].backward(rew[i], terminal=done)
            agent_obs[i] = agents[i].processor.process_observation(obs)

env.close()

for i,a in enumerate(agents):
    a.save_weights("agent_%i_weights.h5f"%(i),overwrite=True)

