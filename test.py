from mae_envs.envs import hide_and_seek
from util import convert_action
from agent import get_agent
import tensorflow as tf
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
rewardWrapper = hide_and_seek.HideAndSeekRewardWrapper(env, n_hiders=hiders, n_seekers=seekers)
trackStatW = hide_and_seek.TrackStatWrapper(env, boxes, ramps, food)

# run one episode
env.seed(42)
env.reset()

agent = get_agent(env)

for t in range(1000):
    env.render()
    print(agent.forward(env))
    a = env.action_space.sample()  # still need to figure out action format.
    action = convert_action(a)
    obs, rew, done, info = env.step(action)  # take a random action + return current state, reward + if episode is done.
    break
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

env.close()

