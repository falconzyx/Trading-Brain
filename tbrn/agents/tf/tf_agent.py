import random
import numpy as np

from tbrn.history import History
from tbrn.memories import ReplayMemory
from tbrn.brains.tf.tf_brain import TFBrain

class Agent(object):
    def __init__(self, config, sess, load, step_op):
        self.config = config
        
        self.history = History(config)
        self.brain = TFBrain(config, sess, load, step_op)
        self.memory = ReplayMemory(config, self.brain.model_dir)
        
    def init(self, state):
        self.brain.init()
        for _ in range(self.config.history_length):
            self.history.add(state)
    
    def predict(self, step, s_t=None, test_ep=None):
        
        #use last state if not set here
        if s_t is None:
            s_t = self.history.get()
            
        #calculate exploratory exponential value for random or decided action
        self.ep = test_ep or (self.config.ep_end + max(0., (self.config.ep_start - self.config.ep_end)
                                * (self.config.ep_end_t - max(0., step - self.config.learn_start)) / self.config.ep_end_t))

        if random.random() < self.ep:
            #use bias from config for random choice
            action = np.random.choice(self.config.action_size, p=self.config.randact)
        else:
            #actual forward pass through network
            action = self.brain.predict(s_t)

        return action
    
    def observe(self, step, state, reward, action, terminal):
        #cap reward
        reward = max(self.config.min_reward, min(self.config.max_reward, reward))
    
        self.history.add(state)
        self.memory.add(state, action, reward, None, terminal)
        
        loss = None
        q_t = None
    
        if step > self.config.learn_start and step > self.config.batch_size*2:
            #do learning with batch
            if step % self.config.train_frequency == 0:
                loss, q_t = self._q_learning_mini_batch(step)
                
            #update target network
            if step % self.config.target_q_update_step == self.config.target_q_update_step - 1:
                self.brain.update_target_q_network()
                
        return loss, q_t

    def _q_learning_mini_batch(self, step):
        if self.memory.count < self.config.history_length:
            return None, None
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
        
        target_q_t = self.brain.calc_target(s_t_plus_1, terminal, reward, self.config.discount, self.config.double_q)
        
        return self.brain.train(s_t, target_q_t, action, step)
    
    def inject_summary(self, tag_dict, step):
        self.brain.inject_summary(tag_dict, step)
    
    def save_model(self, step=None):
        self.brain.save_model(step)
        
    def getLR(self, step):
        return self.brain.getLR(step)
    
    def getEpsilon(self):
        return self.ep

