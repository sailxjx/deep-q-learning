import random
import gym
import numpy as np
import pandas as pd
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import logging

logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s %(message)s')

handler1 = logging.StreamHandler()
handler1.setFormatter(formatter)

handler2 = logging.FileHandler("cartpole.log")
handler2.setFormatter(formatter)

logger.handlers = [handler1, handler2]
logger.setLevel(logging.INFO)

from sum_tree import SumTree, Memory

import time
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K


class PER_D3QNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(10000)
        self.gamma = 0.95  # discount rate, the later the action, the minor the rewards
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # decay exploration rate when episodes accumulated
        self.learning_rate = 0.001
        self.TAU = 0.1

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        input_shape = (self.state_size, )
        X_input = Input(input_shape)
        X = X_input
        X = Dense(24,
                  input_shape=input_shape,
                  activation="relu",
                  kernel_initializer='he_uniform')(X)
        X = Dense(24, activation="relu", kernel_initializer='he_uniform')(X)

        action_space = self.action_size
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                             output_shape=(action_space, ))(state_value)

        action_advantage = Dense(action_space,
                                 kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(
            lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
            output_shape=(action_space, ))(action_advantage)

        X = Add()([state_value, action_advantage])

        model = Model(inputs=X_input, outputs=X)
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # Explore
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)  # Exploit
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        tree_idx, minibatch = self.memory.sample(batch_size)
        indices = np.arange(batch_size, dtype=np.int32)

        # Confusing, read the replies after the post https://keon.github.io/deep-q-learning/
        # The real action of current state use the reward in sample (current reward + model predicted rewards in the future)
        # The opposite action use the rewards predicted by the model

        # Current state
        states = np.array([
            state[0] for state, action, reward, next_state, done in minibatch
        ])
        actions = np.array(
            [action for state, action, reward, next_state, done in minibatch])

        # Real rewards of current step
        rewards = np.array(
            [reward for state, action, reward, next_state, done in minibatch])
        # Next states after current actions
        next_states = np.array([
            next_state[0]
            for state, action, reward, next_state, done in minibatch
        ])
        # Expected highest reward when using the next action, will replace the target reward
        # The most difference between dqn and ddqn
        action_next = np.argmax(self.model.predict(next_states), axis=1)
        target_val = self.target_model.predict(next_states)
        target_next = rewards + self.gamma * target_val[indices, action_next]
        # Set done state to the absolute rewards
        done_indices = np.array(
            [done for state, action, reward, next_state, done in minibatch],
            dtype=np.int)
        np.putmask(target_next, done_indices, rewards)

        target = self.model.predict(states)
        target_old = target.copy()

        # Replace with target_next rewards
        target[indices, actions] = target_next

        history = self.model.fit(states, target, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # PER
        absolute_errors = np.abs(target_old[indices, actions] -
                                 target[indices, actions])
        self.memory.batch_update(tree_idx, absolute_errors)

        return loss, np.average(target_val)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


import gym
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = PER_D3QNAgent(state_size, action_size)
batch_size = 32
EPISODES = 1000
MAX_STEPS = 1000
episode_records = []
total_samples = 0
losses = []
avg_Qs = []

for e in range(EPISODES):
    state = env.reset()
    env._max_episode_steps = 1e10
    state = np.reshape(state, [1, state_size])

    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        reward = reward if not done else -10

        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        total_samples += 1
        state = next_state

        # Train model in every step, make full use of samples
        loss, avg_Q = agent.replay(batch_size)  # Train
        losses.append(loss)
        avg_Qs.append(avg_Q)

        if done or step == MAX_STEPS - 1:
            logger.info(
                "episode: {}/{}, score: {}, e: {:.2}, total samples: {}".
                format(e, EPISODES, step, agent.epsilon, total_samples))
            episode_records.append([step, agent.epsilon, total_samples])
            agent.update_target_model()
            break
