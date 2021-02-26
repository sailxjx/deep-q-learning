# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        # Confusing, read the replies after the post https://keon.github.io/deep-q-learning/
        # The real action of current state use the reward in sample (current reward + model predicted rewards in the future)
        # The opposite action use the rewards predicted by the model

        train_X = np.array([
            state[0] for state, action, reward, next_state, done in minibatch
        ])

        # Only not done memories can use the predicted reward value
        not_done_mask = np.array([
            np.array([1, 1]) * (np.arange(self.action_size) == action) *
            (not done) for state, action, reward, next_state, done in minibatch
        ])
        done_mask = np.array([
            np.array([1, 1]) * (np.arange(self.action_size) == action) * done
            for state, action, reward, next_state, done in minibatch
        ])

        # Real rewards of current step
        rewards = np.array(
            [reward for state, action, reward, next_state, done in minibatch])
        # Next states after current actions
        next_states = np.array([
            next_state[0]
            for state, action, reward, next_state, done in minibatch
        ])
        # Expected highest reward when using the next action, will replace the target reward
        target = rewards + (self.gamma *
                            np.amax(self.model.predict(next_states), axis=1))
        target = np.repeat(target.astype("float32"), action_size).reshape(
            (-1, action_size))  # Make target the same size of target_f (y)

        train_y = self.model.predict(train_X)
        np.putmask(train_y, not_done_mask,
                   target)  # Replace with expected rewards
        np.putmask(
            train_y, done_mask,
            np.repeat(rewards.astype("float32"), action_size).reshape(
                (-1, action_size)))

        history = self.model.fit(train_X, train_y, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    MAX_STEPS = 1000

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for step in range(MAX_STEPS):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if step % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}".format(
                        e, EPISODES, step, loss))

            if done or step == MAX_STEPS - 1:
                print("episode: {}/{}, score: {}, e: {:.2}".format(
                    e, EPISODES, step, agent.epsilon))
                break
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
