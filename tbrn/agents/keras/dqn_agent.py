"""
In this example we demonstrate how to implement a DQN agent and
train it to trade optimally on a periodic price signal.
Training time is short and results are unstable.
Do not hesitate to run several times and/or tweak parameters to get better results.
Inspired from https://github.com/keon/deep-q-learning
"""
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tbrn.base.brain import Brain
from tbrn.base.agent import Agent
from tbrn.base.memory import Memory

class KerasBrain(Brain):
    def __init__(self, state_size, action_size, learning_rate):
        self.brain = Sequential()
        neurons_per_layer = 24
        activation = "relu"
        self.brain.add(Dense(neurons_per_layer,
                        input_dim=state_size,
                        activation=activation))
        self.brain.add(Dense(neurons_per_layer, activation=activation))
        self.brain.add(Dense(action_size, activation='linear'))
        self.brain.compile(loss='mse',optimizer=Adam(lr=learning_rate))
        
    def predict(self, X):
        return self.brain.predict(X)
        
    def train(self, X, y, batch_size, epochs, verbose, w=None):
        return self.brain.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

class VectorMemory(Memory):
    def __init__(self, memory_size, state_size, action_size):
        super(VectorMemory, self).__init__(memory_size)
        self.memory = [None] * memory_size
        self.state_size = state_size
        self.action_size = action_size
    
    def add(self, state, action, reward, next_state, terminal):
        super(VectorMemory, self).add(state, action, reward, next_state, terminal)
        self.memory[self.i] = (state, action, reward, next_state, terminal)

    def sample(self, size):
        """Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, size))
        state_batch = np.concatenate(batch[:,0])\
            .reshape(size,self.state_size)
        action_batch = np.concatenate(batch[:,1])\
            .reshape(size,self.action_size)
        reward_batch = batch[:,2]
        next_state_batch = np.concatenate(batch[:,3])\
            .reshape(size,self.state_size)
        done_batch = batch[:,4]
        # action processing
        action_batch = np.where(action_batch==1)
        return state_batch,action_batch,reward_batch,next_state_batch,done_batch
        
class DQNAgent(Agent):
    def __init__(self,
                state_size,
                action_size,
                episodes,
                game_length,
                memory_size=2000,
                train_interval=100,
                gamma=0.95,
                learning_rate=0.001,
                batch_size=64,
                epsilon_min = 0.01
                ):
        super(DQNAgent, self).__init__(epsilon=1.0)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = VectorMemory(memory_size, state_size, action_size)
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = (self.epsilon-epsilon_min)\
            *train_interval/(episodes*game_length) #linear decrease rate
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

    def observe(self, state, action, reward, next_state, done,warming_up=False):
        """Memory Management and training of the agent
        """
        self.memory.add(state, action, reward, next_state, done)
        
        if (not warming_up) and (self.memory.getIndex() % self.train_interval)==0 :
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state, action, reward, next_state, done = self.memory.sample(self.batch_size)
            reward += (self.gamma
                * np.logical_not(done)
                * np.amax(self.brain.predict(next_state),
                axis=1))
            q_target = self.brain.predict(state)
            q_target[action[0],action[1]] = reward
            return self.brain.train(state,q_target,
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=False)
