import numpy as np
import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Dropout
from keras.optimizers import Adam
from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from collections import OrderedDict

observation_size = 0
ingy = True


def get_agent(env, agent_id, model=1):
    global observation_size
    # Count number of actions
    if not ingy:
        nb_actions = env.action_space['action_movement'][0].shape[0] + 2
        # Count number of observations for input
        if observation_size == 0:
            observation_size += env.observation_space['observation_self'].shape[0]
            observation_size += env.observation_space['agent_qpos_qvel'].shape[0] * \
                                env.observation_space['agent_qpos_qvel'].shape[1]
            observation_size += env.observation_space['box_obs'].shape[0] * env.observation_space['box_obs'].shape[1]
            observation_size += env.observation_space['ramp_obs'].shape[0] * env.observation_space['ramp_obs'].shape[1]
            # TODO: Not sure whether to include mask_a*_obs and mask_ab_obs_spoof in this observation input -AH
    else:
        nb_actions = env.action_space.spaces['action_movement'].spaces[0].shape[0][0] + 2
        # Count number of observations for input
        if observation_size == 0:
            observation_size += env.observation_space.spaces['observation_self'].shape[0]
            if 'lidar' in env.observation_space.spaces:
                observation_size += env.observation_space.spaces['lidar'].shape[0]
            observation_size += env.observation_space.spaces['agent_qpos_qvel'].shape[0] * \
                                env.observation_space.spaces['agent_qpos_qvel'].shape[1]
            observation_size += env.observation_space.spaces['box_obs'].shape[0] * \
                                env.observation_space.spaces['box_obs'].shape[1]
            observation_size += env.observation_space.spaces['ramp_obs'].shape[0] * \
                                env.observation_space.spaces['ramp_obs'].shape[1]

    if model == 1:
        # Build the actor model
        actor = Sequential()
        actor.add(Flatten(input_shape=(1, observation_size,)))
        actor.add(Dense(400))
        actor.add(Activation('relu'))
        actor.add(Dense(300))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('sigmoid'))  # Return values from 0 to 1
        # print(actor.summary())

        # Build the critic model
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1, observation_size,), name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Dense(400)(flattened_observation)
        x = Activation('relu')(x)
        x = Concatenate()([x, action_input])
        x = Dense(300)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        # print(critic.summary())

        # Build the agent
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=2.15, mu=0, sigma=3)
        agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=4000, nb_steps_warmup_actor=4000,
                          random_process=random_process, gamma=.9, target_model_update=1e-3,
                          processor=MujocoProcessor(agent_id))
        agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    elif model == 2:
        # Build the actor model
        actor = Sequential()
        actor.add(Flatten(input_shape=(1, observation_size,)))
        actor.add(Dense(400))
        actor.add(Activation('relu'))
        actor.add(Dense(300))
        actor.add(Dropout(0.3))
        actor.add(Activation('relu'))
        actor.add(Dense(100))
        actor.add(Dropout(0.2))
        actor.add(Activation('elu'))
        actor.add(Dense(50))
        actor.add(Dropout(0.2))
        actor.add(Activation('elu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('softmax'))  # Return values from 0 to 1
        # print(actor.summary())

        # Build the critic model
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1, observation_size,), name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Dense(400)(flattened_observation)
        x = Activation('relu')(x)
        x = Concatenate()([x, action_input])
        x = Dense(300)(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(100)(x)
        x = Activation('elu')(x)
        x = Dropout(0.2)(x)
        x = Dense(50)(x)
        x = Activation('elu')(x)
        x = Dropout(0.2)(x)
        x = Dense(1)(x)
        x = Activation('tanh')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        # print(critic.summary())

        # Build the agent
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=2.8, mu=0, sigma=3.5)
        agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=500, nb_steps_warmup_actor=500,
                          random_process=random_process, gamma=.9, target_model_update=5e-2,
                          processor=MujocoProcessor(agent_id))
        agent.compile([Adam(lr=5e-1, decay=0.9), Adam(lr=5e-1, decay=0.9)], metrics=['mae'])

    return agent

class MujocoProcessor(WhiteningNormalizerProcessor):
    def __init__(self,agent_id):
        self.normalizer = None
        self.agent_id = agent_id

    def process_action(self, action):
        temp = []
        temp.append(int(np.clip(action[0],0,1)*10))
        temp.append(int(np.clip(action[1],0,1)*10))
        temp.append(int(np.clip(action[2],0,1)*10))
        temp.append(int(np.clip(action[3],0,1)))
        temp.append(int(np.clip(action[4],0,1)))
        #for a in action:
        #    temp.append(int(a))
        output = OrderedDict([])
        output['action_movement'] = np.array(temp[0:3])
        output['action_pull'] = np.array(temp[3])
        output['action_glueall'] = np.array(temp[4])
        return output

    def process_observation(self,observation):
        obs = observation['observation_self'][self.agent_id]
        if 'lidar' in observation:
            obs = np.append(obs, observation['lidar'][self.agent_id].flatten())
        obs = np.append(obs,observation['agent_qpos_qvel'][self.agent_id].flatten())
        obs = np.append(obs,observation['box_obs'][self.agent_id].flatten())
        obs = np.append(obs,observation['ramp_obs'][self.agent_id].flatten())
        return obs

