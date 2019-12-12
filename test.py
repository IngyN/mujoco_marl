from mae_envs.envs import hide_and_seek
from util import *
from agent import get_agent
from copy import deepcopy
import numpy as np
import tensorflow as tf
import datetime
import pickle as pkl
from mwu import *
tf.compat.v1.disable_eager_execution()

# environment parameters (there's more). Check make_env
hiders = 1
seekers = 1
boxes = 1
ramps = 1
food = 0
rooms = 2

env = hide_and_seek.make_env(n_hiders=hiders, n_seekers=seekers, n_boxes=boxes, n_ramps=ramps, n_food=food, n_rooms=rooms)

# probably shouldn't use those two. but was testing.
#rewardWrapper = hide_and_seek.HideAndSeekRewardWrapper(env, n_hiders=hiders, n_seekers=seekers)
#trackStatW = hide_and_seek.TrackStatWrapper(env, boxes, ramps, food)

# run one episode
env.seed(42)

load_weights = False
display = False
agents = []
for i in range(hiders+seekers):
    agents.append(MWU())
    #agents.append(get_agent(env,i))
    if load_weights:
        f = open("mwu_%i.data"%(i),'rb')
        w = pkl.load(f)
        f.close()
        agents[-1].set_weights(w)
        #agents[-1].load_weights("agent_%i_weights.h5f"%(i))

#https://github.com/keras-rl/keras-rl/blob/master/rl/core.py
obs = None
rew = None
done = None
info = None
agent_obs = [[]]*(hiders+seekers)
agent_act = [[]]*(hiders+seekers)
qqq = np.zeros(hiders+seekers)
if display:
    env.render()
#if not load_weights:
#    for a in agents:
#        a.training = True

print("    "+str(datetime.datetime.now()))
for e in range(10000): #num_episodes
    #env.reset()
    if (e+1)%100 == 0:
        print(str(e+1)+" "+str(datetime.datetime.now())+" "+str(qqq/100))
        qqq = np.zeros(hiders+seekers)
    for t in range(1000):
        if obs is None:
            obs = deepcopy(env.reset())
            rew = np.float32(0)
            #for i in range(hiders+seekers):
            #    agent_obs[i] = agents[i].processor.process_observation(obs)
        for i in range(hiders+seekers):
            agent_act[i] = agents[i].sample()
            #action_i = agents[i].forward(agent_obs[i])
            #if i < hiders:
            #    #agent_act[i] = random_action()
            #    agent_act[i] = no_action()
            #else:
            #    agent_act[i] = agents[i].processor.process_action(action_i)
            #agent_act[i] = agents[i].processor.process_action(action_i)
            done = False
        action = convert_action(agent_act)
        obs, rew, done, info = env.step(action)  # take a random action + return current state, reward + if episode is done.
        obs = deepcopy(obs)
        qqq+=rew
        if display:
            env.render()
        for i in range(hiders+seekers):
            agents[i].backward(agent_act[i],rew[i])
            #metrics = agents[i].backward(rew[i], terminal=done)
            #agent_obs[i] = agents[i].processor.process_observation(obs)
        if done:
            #for i in range(hiders+seekers):
            #    agents[i].forward(agent_obs[i])
            #    agents[i].backward(0., terminal=False)
            obs = None
            break
    #if (e+1)%10000 == 0:
    #    for i,a in enumerate(agents):
    #        a.save_weights("%i/agent_%i_weights.h5f"%(e+1,i),overwrite=True)

env.close()

for i,a in enumerate(agents):
    f = open("mwu_%i.data"%(i),'wb')
    pkl.dump(a.weights,f)
    f.close()
    print(a.weights)

#if not load_weights:
#    for i,a in enumerate(agents):
#        a.save_weights("agent_%i_weights.h5f"%(i),overwrite=True)

