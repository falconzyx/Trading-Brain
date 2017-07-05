#import os
#import time
import random
import numpy as np
import tensorflow as tf

from tbrn.model_saver import ModelSaver
from tbrn.history import History
from tbrn.memories import ReplayMemory
from tbrn.base.brain import Brain
#from .ops import linear, conv2d, clipped_error
#from utils import get_time, save_pkl, load_pkl

class TFBrain(Brain):
    def __init__(self, env_name, history_length, input_size, num_hidden, action_size, dueling, learning_rate_params,
                 model_dir, sess):
        self.w = {}
        self.t_w = {}
        self.history_length = history_length
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.action_size = action_size
        self.dueling = dueling
        self.learning_rate_params = learning_rate_params
        self.env_name = env_name
        self.model_dir = model_dir
        self.sess = sess
        activation_fn = tf.nn.relu
    
        # training network
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('int32',
                [None, self.history_length, self.input_size], name='s_t')
            self.s_t_flat = tf.reshape(self.s_t, [-1,self.input_size*self.history_length])
            self.l1, self.w['l1_w'], self.w['l1_b'] = self.nn_layer(self.s_t_flat, self.num_hidden, 'l1', activation_fn)
            #self.l2, self.w['l2_w'], self.w['l2_b'] = self.nn_layer(self.l1, self.num_hidden, 'l2', activation_fn)
        
            if self.dueling:
                self.value_hid, self.w['l2_val_w'], self.w['l2_val_b'] = \
                    self.nn_layer(self.l1, self.num_hidden, 'value_hid', activation_fn)
        
                self.adv_hid, self.w['l2_adv_w'], self.w['l2_adv_b'] = \
                    self.nn_layer(self.l1, self.num_hidden, 'adv_hid', activation_fn)
        
                self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                    self.nn_layer(self.value_hid, 1, 'value_out')
        
                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                    self.nn_layer(self.adv_hid, self.action_size, 'adv_out')
        
                # Average Dueling
                self.q = self.value + (self.advantage - 
                  tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            else:            
                self.q, self.w['q_w'], self.w['q_b'] = self.nn_layer(self.l1, self.action_size, name='q')
            
            self.q_action = tf.argmax(self.q, dimension=1)
            
            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.action_size):
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')
    
        # target network
        with tf.variable_scope('target'):          
            self.target_s_t = tf.placeholder('int32', 
                [None, self.history_length, self.input_size], name='target_s_t')
            self.target_s_t_flat = tf.reshape(self.target_s_t, [-1,self.input_size*self.history_length])
            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = self.nn_layer(self.target_s_t_flat, self.num_hidden,
                                                                               'target_l1', activation_fn) 
            #self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = self.nn_layer(self.l1, self.num_hidden,
            #                                                                   'target_l2', activation_fn) 
            
            if self.dueling:
                self.t_value_hid, self.t_w['l2_val_w'], self.t_w['l2_val_b'] = \
                    self.nn_layer(self.target_l1, self.num_hidden, 'target_value_hid', activation_fn)
        
                self.t_adv_hid, self.t_w['l2_adv_w'], self.t_w['l2_adv_b'] = \
                    self.nn_layer(self.target_l1, self.num_hidden, 'target_adv_hid', activation_fn)
        
                self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
                    self.nn_layer(self.t_value_hid, 1, 'target_value_out')
        
                self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
                    self.nn_layer(self.t_adv_hid, self.action_size, 'target_adv_out')
        
                # Average Dueling
                self.target_q = self.t_value + (self.t_advantage - 
                                                tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
            else:                
                self.target_q, self.t_w['q_w'], self.t_w['q_b'] = self.nn_layer(self.target_l1,
                                                                            self.action_size, name='target_q')
            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)
    
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
    
            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])
    
        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')
            
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
            
            delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            
            self.loss = tf.reduce_mean(TFBrain._clipped_error(delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_params['min'],
                  tf.train.exponential_decay(
                      self.learning_rate_params['lr'],
                      self.learning_rate_step,
                      self.learning_rate_params['decay_step'],
                      self.learning_rate_params['decay'],
                      staircase=True))
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op,
                                                   momentum=0.95, epsilon=0.01).minimize(self.loss)
    
        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', 'episode.max reward',
                                   'episode.min reward', 'episode.avg reward', 'episode.num of game',
                                   'training.learning_rate']
    
            self.summary_placeholders = {}
            self.summary_ops = {}
            
            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.scalar("%s-%s" % (self.env_name, tag), self.summary_placeholders[tag])
    
            histogram_summary_tags = ['episode.rewards', 'episode.actions']
            
            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])
                
            self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)
        
    def init(self):
        self.update_target_q_network()
        
    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})
            
    def calc_target(self, s_t_plus_1, terminal, reward, discount, double_q):
        
        if double_q:
            # Double Q-learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
    
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                    self.target_s_t: s_t_plus_1,
                    self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
                })
            target_q_t = (1. - terminal) * discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
    
            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward
            
        return target_q_t
    
    #def learn(self, target_q_t, action, s_t, step):
    def train(self, s_t, target_q_t, action, step):
        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
                self.target_q_t: target_q_t,
                self.action: action,
                self.s_t: s_t,
                self.learning_rate_step: step,
            })
    
        self.writer.add_summary(summary_str, step)
        return loss, q_t
    
    #def forward(self, s_t, getVal=False):
    def predict(self, s_t, getVal=False):
        if not getVal:
            action = self.q_action.eval({self.s_t: [s_t]})[0]
            return action
        else:
            vals = np.empty(shape=(s_t.shape[0],self.action_size))
            for i,item in enumerate(s_t):
                val = self.sess.run([self.q],{self.s_t:[[item]]})
                vals[i,:]=val[0][0]
            return vals
    
    def getLR(self, step):
        return self.learning_rate_op.eval({self.learning_rate_step: step})
    
    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, step)
        
    @staticmethod
    def _weight_init():
        return tf.truncated_normal_initializer(0, 0.1)

    @staticmethod
    def _bias_init():
        return tf.constant_initializer(0.01)
        
    @staticmethod
    def nn_layer(input_tensor, output_dim, name, act=None):
        shape = input_tensor.get_shape().as_list()
        with tf.variable_scope(name):
            w = tf.get_variable('Matrix', [shape[1], output_dim], tf.float32, initializer=TFBrain._weight_init())
            b = tf.get_variable('bias', [output_dim], initializer=TFBrain._bias_init())

            out = tf.nn.bias_add(tf.matmul(tf.to_float(input_tensor), w), b)
        
            if act != None:
                return act(out), w, b
            else:
                return out, w, b

    @staticmethod
    def _clipped_error(x):
        # Huber loss
        try:
            return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        except:
            return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class Agent(ModelSaver):
    def __init__(self, config, sess):
        super(Agent, self).__init__(config)
        self._qs = []
                
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config, self.model_dir)
        
        self.brain = TFBrain(self.env_name, self.history_length, self.input_size, self.num_hidden, self.action_size, self.dueling,
                           self.learning_rate_params, self.model_dir, sess)
        
    def init(self, state):
        self.brain.init()
        for _ in range(self.history_length):
            self.history.add(state)
            
    def sampleQS(self, testSet):
        qv = self._getQ(testSet)
        self._qs.append(qv)
        return qv
    
    def getQS(self):
        return self._qs
    
    def _getQ(self, states):
        action_values = self.brain.predict(states, True)
        return action_values
            
    def inject_summary(self, tag_dict, step):
        self.brain.inject_summary(tag_dict, step)
        
    def getLR(self, step):
        return self.brain.getLR(step)
    
    def _random_action(self):
        return np.random.choice(self.action_size, p=self.randact)
    
    def predict(self, step, s_t=None, test_ep=None):
        if s_t is None:
            s_t = self.history.get()
        self.ep = test_ep or (self.ep_end + max(0., (self.ep_start - self.ep_end)
                                           * (self.ep_end_t - max(0., step - self.learn_start)) / self.ep_end_t))

        if random.random() < self.ep:
            action = self._random_action()
        else:
            action = self.brain.predict(s_t)

        return action
    
    def getEpsilon(self):
        return self.ep

    def observe(self, step, state, reward, action, terminal):
        reward = max(self.min_reward, min(self.max_reward, reward))
    
        self.history.add(state)
        self.memory.add(state, action, reward, None, terminal)
        
        loss = None
        q_t = None
    
        if step > self.learn_start and step > self.batch_size*2:
            if step % self.train_frequency == 0:
                loss, q_t = self.q_learning_mini_batch(step)
    
            if step % self.target_q_update_step == self.target_q_update_step - 1:
                self.brain.update_target_q_network()
                
        return loss, q_t

    def q_learning_mini_batch(self, step):
        if self.memory.count < self.history_length:
            return None, None
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
        
        target_q_t = self.brain.calc_target(s_t_plus_1, terminal, reward, self.discount, self.double_q)
        
        return self.brain.train(s_t, target_q_t, action, step)

    def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
        if test_ep == None:
            test_ep = self.ep_end
