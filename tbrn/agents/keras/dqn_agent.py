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
from tbrn.memories import VectorMemory
import tensorflow as tf

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
        
    def train(self, X, y, batch_size, epochs, verbose, step=None, action=None):
        vals = self.brain.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)
        return vals.history['loss'][0]
    
class TFBrain(Brain):
    def __init__(self, state_size, action_size, learning_rate_params, env_name):
        self.env_name = env_name
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate_params = learning_rate_params
        self.w = {}
        self.t_w = {}
        self.num_hidden = 12
        self._init = False
        activation_fn = tf.nn.relu
        self.dueling = True
        self.sess = tf.Session()
        
        # training network
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('int32', [None, self.state_size], name='s_t')
            self.l1, self.w['l1_w'], self.w['l1_b'] = TFBrain._nn_layer(self.s_t, self.num_hidden, 'l1', activation_fn)
            if self.dueling:
                self.value_hid, self.w['l2_val_w'], self.w['l2_val_b'] = \
                    TFBrain._nn_layer(self.l1, self.num_hidden, 'value_hid', activation_fn)
        
                self.adv_hid, self.w['l2_adv_w'], self.w['l2_adv_b'] = \
                    TFBrain._nn_layer(self.l1, self.num_hidden, 'adv_hid', activation_fn)
        
                self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                    TFBrain._nn_layer(self.value_hid, 1, 'value_out')
        
                self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
                    TFBrain._nn_layer(self.adv_hid, self.action_size, 'adv_out')
        
                # Average Dueling
                self.q = self.value + (self.advantage - 
                  tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
            else:
                self.l2, self.w['l2_w'], self.w['l2_b'] = TFBrain._nn_layer(self.l1, self.num_hidden, 'l2', activation_fn)
                self.q, self.w['q_w'], self.w['q_b'] = TFBrain._nn_layer(self.l2, self.action_size, name='q')
            self.q_action = tf.argmax(self.q, dimension=1)
            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.action_size):
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        # target network
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('int32', 
                [None, self.state_size], name='target_s_t')
            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = TFBrain._nn_layer(self.target_s_t, self.num_hidden,
                                                                               'target_l1', activation_fn)
            if self.dueling:
                self.t_value_hid, self.t_w['l2_val_w'], self.t_w['l2_val_b'] = \
                    TFBrain._nn_layer(self.target_l1, self.num_hidden, 'target_value_hid', activation_fn)
        
                self.t_adv_hid, self.t_w['l2_adv_w'], self.t_w['l2_adv_b'] = \
                    TFBrain._nn_layer(self.target_l1, self.num_hidden, 'target_adv_hid', activation_fn)
        
                self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
                    TFBrain._nn_layer(self.t_value_hid, 1, 'target_value_out')
        
                self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
                    TFBrain._nn_layer(self.t_adv_hid, self.action_size, 'target_adv_out')
        
                # Average Dueling
                self.target_q = self.t_value + (self.t_advantage - 
                                                tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
            else:
                self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = TFBrain._nn_layer(self.target_l1, self.num_hidden,
                                                                                   'target_l2', activation_fn)
                self.target_q, self.t_w['q_w'], self.t_w['q_b'] = TFBrain._nn_layer(self.target_l2,
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
            #self.target_q_t = tf.placeholder('float32', [None, self.action_size], name='target_q_t')
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')
            
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
            #self.q_acted = tf.placeholder('float32',[None, self.action_size],name='q_acted')
            
            delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            
            self.loss = tf.reduce_mean(TFBrain._clipped_error(delta), name='loss')
            #self.loss = tf.reduce_mean(delta, name='loss')
            #self.loss = tf.reduce_mean(tf.squared_difference(self.target_q_t, self.q, 'loss'))
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_params['min'],
                  tf.train.exponential_decay(
                      self.learning_rate_params['lr'],
                      self.learning_rate_step,
                      self.learning_rate_params['decay_step'],
                      self.learning_rate_params['decay'],
                      staircase=True))
            self.optim = tf.train.RMSPropOptimizer(0.001,#self.learning_rate_op,
                                                   momentum=0.95, epsilon=0.01).minimize(self.loss)
            #self.optim = tf.train.AdamOptimizer(self.learning_rate_op).minimize(self.loss)
            #self.optim = tf.train.AdamOptimizer(0.001).minimize(self.loss)
    
        '''with tf.variable_scope('summary'):
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
                
            self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)'''
        
        '''self.brain = Sequential()
        neurons_per_layer = 24
        activation = "relu"
        self.brain.add(Dense(neurons_per_layer,
                        input_dim=state_size,
                        activation=activation))
        self.brain.add(Dense(neurons_per_layer, activation=activation))
        self.brain.add(Dense(action_size, activation='linear'))
        self.brain.compile(loss='mse',optimizer=Adam(lr=learning_rate))'''
        
    def predict(self, X):
        with self.sess.as_default():
            if not self._init:
                tf.global_variables_initializer().run()
                self.update_target_q_network()
                self._init = True            
            #action = self.q_action.eval({self.s_t: X})[0]
            action = self.q.eval({self.s_t: X})
            return action
        
    def train(self, X, y, batch_size, epochs, verbose, w=None, step=None, action=None):
        #return self.brain.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)
        #loss.history
        #{'loss': [0.084096193313598633]}
        loss, _ = self.sess.run([self.loss, self.q], {
        #loss = self.sess.run([self.loss], {
            self.target_q_t: y,
            self.action: action,
            self.s_t: X,
            self.learning_rate_step: step
        })
    
        #self.writer.add_summary(summary_str, step)
        #return loss, q_t
        return loss
    
    def update_target_q_network(self):
        with self.sess.as_default():
            for name in self.w.keys():
                self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def _calc_target(self, s_t_plus_1, state, terminal, reward, discount, double_q):
        with self.sess.as_default():
            if not self._init:
                tf.global_variables_initializer().run()
                self.update_target_q_network()
                self._init = True
            if double_q:
                # Double Q-learning
                pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
                #pred_action = self.q_action.eval({self.s_t: state})
        
                q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                        self.target_s_t: s_t_plus_1,
                        self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
                    })
                target_q_t = (1. - terminal) * discount * q_t_plus_1_with_pred_action + reward
            else:
                q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
        
                terminal = np.array(terminal) + 0.
                max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
                target_q_t = (1. - terminal) * discount * max_q_t_plus_1 + reward
                
            return target_q_t
    
    @staticmethod
    def _weight_init():
        return tf.truncated_normal_initializer(0, 0.1)

    @staticmethod
    def _bias_init():
        return tf.constant_initializer(0.01)
        
    @staticmethod
    def _nn_layer(input_tensor, output_dim, name, act=None):
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

class DQNAgent(Agent):
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
        super(DQNAgent, self).__init__(epsilon=1.0)
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
        self.step = 0
        self.double_q = True
        extra = {'env_name':'Trading'}
        #learning_rate_params = {'lr':0.002, 'min':0.00025, 'decay':0.96, 'decay_step':episode_length/100}
        learning_rate_params = {'lr':0.002, 'min':0.00025, 'decay':0.96, 'decay_step':150}
        self.brain = TFBrain(state_size, action_size, learning_rate_params, *extra)
        #self.brain = KerasBrain(state_size, action_size, learning_rate)

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
        '''reward += (self.gamma
            * np.logical_not(done)
            * np.amax(self.brain.predict(next_state),
            axis=1))
        q_target = self.brain.predict(state)
        q_target[action[0],action[1]] = reward
        return q_target'''
        return self.brain._calc_target(next_state, state, done, reward, self.gamma, self.double_q)

    def observe(self, state, action, reward, next_state, done,warming_up=False):
        """Memory Management and training of the agent
        """
        self.memory.add(state, action, reward, next_state, done)
        self.step = self.step + 1
        
        if (not warming_up) and (self.memory.getIndex() % self.train_interval)==0 :
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state, action, reward, next_state, done = self.memory.sample(self.batch_size)
            q_target = self._calcTarget(next_state, state, action, reward, done)
            return self.brain.train(state,q_target,
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=False,
                                  step=self.step,
                                  action=action[1])
