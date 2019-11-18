
from mae_envs.envs import hide_and_seek
from util import convert_action

# environment parameters (there's more). Check make_env
n_hiders = 2
n_seekers = 1
boxes = 1
ramps = 1
food = 0
rooms = 2



env = hide_and_seek.make_env(n_hiders = n_hiders, n_seekers=n_seekers, n_boxes=boxes, n_ramps=ramps, n_food= 0, n_rooms=2)

# probably shouldn't use those two. but was testing.
rewardWrapper = hide_and_seek.HideAndSeekRewardWrapper(env, n_hiders=n_hiders, n_seekers=n_seekers)
trackStatW = hide_and_seek.TrackStatWrapper(env, boxes, ramps, food)

# run one episode

env.reset()

for t in range(1000):
    env.render()
    a = env.action_space.sample()  # still need to figure out action format.
    action = convert_action(a)
    obs, rew, done, info = env.step(action)  # take a random action + return current state, reward + if episode is done.
    print(rew)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

env.close()




