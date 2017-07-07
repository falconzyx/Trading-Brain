"""
In this example we demonstrate how to implement a DQN agent and
train it to trade optimally on a periodic price signal.
Training time is short and results are unstable.
Do not hesitate to run several times and/or tweak parameters to get better results.
Inspired from https://github.com/keon/deep-q-learning
"""
import numpy as np
import random
from tbrn.base.agent import Agent
from tbrn.memories import VectorMemory
from tbrn.brains.keras.keras_brain import KerasBrain

class KerasAgent(Agent):
    def __init__(self,
                state_size,
                action_size,
                episodes,
                episode_length,
                memory_size=2000,
                train_interval=100,
                gamma=0.95,
                learning_rate=0.001,
                batch_size=64,
                epsilon_min = 0.01
                ):
        super(KerasAgent, self).__init__(epsilon=1.0)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = VectorMemory(memory_size, state_size, action_size)
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = (self.epsilon-epsilon_min)\
            *train_interval/(episodes*episode_length) #linear decrease rate
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.brain = KerasBrain(state_size, action_size, learning_rate)

    def act(self, state):
        """Acting Policy of the DQNAgent
        """
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon:
            action[random.randrange(self.action_size)]=1
        else:
            state = state.reshape(1,self.state_size)
            act_values = self.brain.predict(state)
            action[np.argmax(act_values[0])] = 1
        return action
    
    def _calcTarget(self, next_state, state, action, reward, done):
        reward += (self.gamma
            * np.logical_not(done)
            * np.amax(self.brain.predict(next_state),
            axis=1))
        q_target = self.brain.predict(state)
        q_target[action[0],action[1]] = reward
        return q_target

    def observe(self, state, action, reward, next_state, done,warming_up=False):
        """Memory Management and training of the agent
        """
        self.memory.add(state, action, reward, next_state, done)
        
        if (not warming_up) and (self.memory.getIndex() % self.train_interval)==0 :
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state, action, reward, next_state, done = self.memory.sample(self.batch_size)
            q_target = self._calcTarget(next_state, state, action, reward, done)
            return self.brain.train(state,q_target,
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=False)
