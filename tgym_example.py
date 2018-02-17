"""
In this example we demonstrate how to implement a DQN agent and
train it to trade optimally on a periodic price signal.
Training time is short and results are unstable.
Do not hesitate to run several times and/or tweak parameters to get better results.
Inspired from https://github.com/keon/deep-q-learning
"""
import random

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from _TYPE_StochasticProcess import ExampleOUProcess
from _TYPE_OptimalTrader import UnivariateKerasTrader, OptimalTrader
from _TYPE_Simulator import Simulator


class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 training_size=100000,
                 memory_size=2000,
                 train_interval=100,
                 gamma=0.95,
                 learning_rate=0.001,
                 batch_size=64,
                 epsilon_min=0.01
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = (self.epsilon - epsilon_min)\
            * train_interval / training_size  # linear decrease rate
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.brain = self._build_brain()
        self.i = 0

    def _build_brain(self):
        """Build the agent's brain
        """
        brain = Sequential()
        neurons_per_layer = 24
        activation = "relu"
        brain.add(Dense(neurons_per_layer,
                        input_dim=self.state_size,
                        activation=activation))
        brain.add(Dense(neurons_per_layer, activation=activation))
        brain.add(Dense(self.action_size, activation='linear'))
        brain.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return brain

    def act(self, state):
        """Acting Policy of the DQNAgent
        """
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon:
            action[random.randrange(self.action_size)] = 1
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.brain.predict(state)
            action[np.argmax(act_values[0])] = 1
        return action

    def observe(self, state, action, reward, next_state, done, warming_up=False):
        """Memory Management and training of the agent
        """
        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (not warming_up) and (self.i % self.train_interval) == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state, action, reward, next_state, done = self._get_batches()
            reward += (self.gamma
                       * np.logical_not(done)
                       * np.amax(self.brain.predict(next_state),
                                 axis=1))
            q_target = self.brain.predict(state)
            q_target[action[0], action[1]] = reward
            return self.brain.fit(state, q_target,
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=False)

    def _get_batches(self):
        """Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, self.batch_size))
        state_batch = np.concatenate(batch[:, 0])\
            .reshape(self.batch_size, self.state_size)
        action_batch = np.concatenate(batch[:, 1])\
            .reshape(self.batch_size, self.action_size)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3])\
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        # action processing
        action_batch = np.where(action_batch == 1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


class Trader(OptimalTrader):
    def __init__(self, N=10000):
        super().__init__()
        memory_size = 3000
        state_size = 2
        gamma = 0.96
        epsilon_min = 0.01
        batch_size = 64
        action_size = 3
        train_interval = 10
        learning_rate = 0.001
        self.agent = DQNAgent(state_size=state_size,
                         action_size=action_size,
                         memory_size=memory_size,
                         training_size=N,
                         train_interval=train_interval,
                         gamma=gamma,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         epsilon_min=epsilon_min)

    def choose(self, price):
        self.price = price
        self.action = self.agent.act(np.array([price, self.position]))
        new_position = np.where(self.action)[0][0] - self.max_position
        return self.position, new_position - self.position

    def update(self, new_price, new_position, reward):
        state = np.array([self.price, self.position])
        new_state = np.array([new_price, new_position])
        loss = self.agent.observe(state, self.action, reward, next_state, False)

    def _trade(self, price):
        new_position = np.argmax(self.q_map.q_map.predict(np.array([[price, self.position]]))) - self.max_position
        action = new_position - self.position
        self.position = new_position
        return action


if __name__ == "__main__":
    N = 100000
    process_type = ExampleOUProcess
    trader = Trader(N)
    sim = Simulator(process_type(), trader)

    # Instantiating the agent

    # Warming up the agent
    for _ in range(memory_size):
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        agent.observe(state, action, reward, next_state, done, warming_up=True)
    # Training the agent
    for ep in range(episodes):
        state = environment.reset()
        rew = 0
        for _ in range(episode_length):
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            loss = agent.observe(state, action, reward, next_state, done)
            state = next_state
            rew += reward
        print("Ep:" + str(ep)
              + "| rew:" + str(round(rew, 2))
              + "| eps:" + str(round(agent.epsilon, 2))
              + "| loss:" + str(round(loss.history["loss"][0], 4)))
    # Running the agent
    done = False
    state = environment.reset()
    while not done:
        action = agent.act(state)
        state, _, done, info = environment.step(action)
        if 'status' in info and info['status'] == 'Closed plot':
            done = True
        else:
            environment.render()