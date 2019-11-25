import numpy as np
import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

def get_agent(env):
    # Count number of actions
    nb_actions = env.action_space['action_movement'][0].shape[0]+2
    # Count number of observations for input
    observation_size = 0
    observation_size+=env.observation_space['observation_self'].shape[0]
    observation_size+=env.observation_space['agent_qpos_qvel'].shape[0]*env.observation_space['agent_qpos_qvel'].shape[1]
    observation_size+=env.observation_space['box_obs'].shape[0]*env.observation_space['box_obs'].shape[1]
    observation_size+=env.observation_space['ramp_obs'].shape[0]*env.observation_space['ramp_obs'].shape[1]
    #TODO: Not sure whether to include mask_a*_obs and mask_ab_obs_spoof in this observation input -AH

    # Build the actor model
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,observation_size,)))
    actor.add(Dense(400))
    actor.add(Activation('relu'))
    actor.add(Dense(300))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))
    #print(actor.summary())

    # Build the critic model
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,observation_size,), name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Dense(400)(flattened_observation)
    x = Activation('relu')(x)
    x = Concatenate()([x, action_input])
    x = Dense(300)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    #print(critic.summary())

    # Build the agent
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3,
                      processor=MujocoProcessor())
    agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])
    return agent

class MujocoProcessor():
    def process_action(self, action):
        return action
    def process_state_batch(self,batch):
        print(batch) #TODO: Not sure what to do in this function -AH
        return batch

