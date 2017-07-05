import random
import numpy as np
import os
from tbrn.base.memory import Memory
from tbrn.utils import save_npy, load_npy

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
    
"""Code below from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

class ReplayMemory(Memory):
    def __init__(self, config, model_dir):
        super(ReplayMemory, self).__init__(config.memory_size)
        self.model_dir = model_dir
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float32)
        self.states = np.empty((self.memory_size, config.input_size), dtype=np.int32)
        self.terminals = np.empty(self.memory_size, dtype = np.bool)
        self.history_length = config.history_length
        self.dims = config.input_size
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0
    
        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length, self.dims), dtype = np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length, self.dims), dtype = np.float16)

    def add(self, state, action, reward, next_state, terminal):
        #assert state.shape == self.dims
        assert len(state) == self.dims
        # NB! state is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states[self.current,...] = state
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.states[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.states[indexes, ...]

    def sample(self, size=None):
        if size is None:
            size = self.batch_size
        # memory must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < size:
            # find random index 
            while True:
                # sample one index (ignore states wraping over 
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break
            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)
    
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
    
        return self.prestates, actions, rewards, self.poststates, terminals

    def save(self):
        for idx, (name, array) in enumerate(
            zip(['actions', 'rewards', 'states', 'terminals', 'prestates', 'poststates'],
                [self.actions, self.rewards, self.states, self.terminals, self.prestates, self.poststates])):
            save_npy(array, os.path.join(self.model_dir, name))

    def load(self):
        for idx, (name, array) in enumerate(
            zip(['actions', 'rewards', 'states', 'terminals', 'prestates', 'poststates'],
                [self.actions, self.rewards, self.states, self.terminals, self.prestates, self.poststates])):
            array = load_npy(os.path.join(self.model_dir, name))
