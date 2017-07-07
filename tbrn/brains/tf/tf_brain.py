import numpy as np
import tensorflow as tf

from tbrn.model_saver import ModelSaver
from tbrn.base.brain import Brain

#Optional ModelSaver derived - can save and load the network from disk
class TFBrain(Brain, ModelSaver):
    def __init__(self, config, sess, load, step_op):
        super(TFBrain, self).__init__(config, sess)
        
        #network variables set to blank for _createNet
        self.w, self.s_t, self.s_t_flat, self.l1, self.value_hid, self.adv_hid = {}, {}, {}, {}, {}, {}
        self.value, self.advantage, self.q, self.l2 = {}, {}, {}, {}
        #The one tensorflow session
        self.sess = sess
        
        # training network
        with tf.variable_scope('prediction'):
            self._createNet('p')
            
            #best action is the max of the q values
            self.q_action = tf.argmax(self.q['p'], dimension=1)
            
            #for tensorboard - optional
            q_summary = []
            avg_q = tf.reduce_mean(self.q['p'], 0)
            for idx in range(self.action_size):
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')
    
        # target network
        with tf.variable_scope('target'):
            self._createNet('target')
            #need index with the target q for double dqn
            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.q['target'], self.target_q_idx)
    
        #how to copy from prediction network to identical target network
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
    
            for name in self.w['p'].keys():
                self.t_w_input[name] = tf.placeholder('float32', self.w['target'][name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.w['target'][name].assign(self.t_w_input[name])
    
        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')
            
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q['p'] * action_one_hot, reduction_indices=1, name='q_acted')
            
            delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            #almost MSE loss, but using Huber function
            self.loss = tf.reduce_mean(TFBrain._clipped_error(delta), name='loss')
            #exponential decay of learning rate
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_params['min'],
                  tf.train.exponential_decay(
                      self.learning_rate_params['lr'],
                      self.learning_rate_step,
                      self.learning_rate_params['decay_step'],
                      self.learning_rate_params['decay'],
                      staircase=True))
            #RMS/Adam both seem to work
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op,
                                                   momentum=0.95, epsilon=0.01).minimize(self.loss)
    
        #optional tensorboard summary
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

        #initialize the variables, create saver, and load model if load is true
        tf.global_variables_initializer().run()
        self._saver = tf.train.Saver(self.w['p'].values() + [step_op], max_to_keep=30)
        if load:
            self.load_model()
        
    #creates the actual network topology, both for prediction network and target network
    def _createNet(self, name):
        if name not in self.w:
            self.w[name] = {}
        activation_fn = tf.nn.relu
        self.s_t[name] = tf.placeholder('int32',
            [None, self.history_length, self.input_size], name='s_t.{0}'.format(name))
        self.s_t_flat[name] = tf.reshape(self.s_t[name], [-1,self.input_size*self.history_length])
        self.l1[name], self.w[name]['l1_w'], self.w[name]['l1_b'] = self.nn_layer(self.s_t_flat[name], self.num_hidden,
                                                                                  'l1.{0}'.format(name), activation_fn)
    
        if self.dueling:
            self.value_hid[name], self.w[name]['l2_val_w'], self.w[name]['l2_val_b'] = \
                self.nn_layer(self.l1[name], self.num_hidden, 'value_hid.{0}'.format(name), activation_fn)
    
            self.adv_hid[name], self.w[name]['l2_adv_w'], self.w[name]['l2_adv_b'] = \
                self.nn_layer(self.l1[name], self.num_hidden, 'adv_hid.{0}'.format(name), activation_fn)
    
            self.value[name], self.w[name]['val_w_out'], self.w[name]['val_w_b'] = \
                self.nn_layer(self.value_hid[name], 1, 'value_out.{0}'.format(name))
    
            self.advantage[name], self.w[name]['adv_w_out'], self.w[name]['adv_w_b'] = \
                self.nn_layer(self.adv_hid[name], self.action_size, 'adv_out.{0}'.format(name))
    
            # Average Dueling
            self.q[name] = self.value[name] + (self.advantage[name]
                                               - tf.reduce_mean(self.advantage[name], reduction_indices=1, keep_dims=True))
        else:
            self.l2[name], self.w[name]['l2_w'], self.w[name]['l2_b'] = self.nn_layer(self.l1[name], self.num_hidden,
                                                                                      'l2.{0}'.format(name), activation_fn)
            self.q[name], self.w[name]['q_w'], self.w[name]['q_b'] = self.nn_layer(self.l2[name], self.action_size,
                                                                                   name='q.{0}'.format(name))
        
    def init(self):
        self.update_target_q_network()
        
    def update_target_q_network(self):
        for name in self.w['p'].keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w['p'][name].eval()})
            
    def calc_target(self, s_t_plus_1, terminal, reward, discount, double_q):
        #bellman equation here
        if double_q:
            # Double Q-learning
            pred_action = self.q_action.eval({self.s_t['p']: s_t_plus_1})
    
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                    self.s_t['target']: s_t_plus_1,
                    self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
                })
            target_q_t = (1. - terminal) * discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.q['target'].eval({self.s_t['target']: s_t_plus_1})
    
            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward
            
        return target_q_t

    def train(self, s_t, target_q_t, action, step):
        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q['p'], self.loss, self.q_summary], {
                self.target_q_t: target_q_t,
                self.action: action,
                self.s_t['p']: s_t,
                self.learning_rate_step: step,
            })
    
        self.writer.add_summary(summary_str, step)
        return loss, q_t

    def predict(self, s_t, getVal=False):
        if not getVal: #normal route here, just get the best action
            action = self.q_action.eval({self.s_t['p']: [s_t]})[0]
            return action
        else: # for getting q values, iterate through and accumulate
            vals = np.empty(shape=(s_t.shape[0],self.action_size))
            for i,item in enumerate(s_t):
                val = self.sess.run([self.q['p']],{self.s_t['p']:[[item]]})
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
        return tf.truncated_normal_initializer(0, 0.2)

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
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
