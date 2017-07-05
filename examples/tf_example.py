import random
import argparse
import os
#import time
from tqdm import tqdm
#from agent import Agent
from tbrn.agents.tf.tf_agent import Agent
from tbrn.trading_game import TradingGame
from tbrn.config import get_config
import tensorflow as tf
import numpy as np
import pickle

# Set random seed
tf.set_random_seed(123)
random.seed(123)

class Stats(object):
    def __init__(self, test_step):
        self.test_step = test_step
        self.stats = None
        self.clear()
    def clear(self):
        self.num_game, self.update_count, self.ep_reward = 0, 0, 0.
        self.total_reward, self.total_loss, self.total_q = 0., 0., 0.
        self.max_avg_ep_reward = 0
        self.ep_rewards, self.actions = [], []
    def update(self, loss, q_t, reward, action):
        if loss is not None:
            self.total_loss += loss
            self.total_q += q_t.mean()
            self.update_count += 1
        self.ep_reward += reward
        self.actions.append(action)
        self.total_reward += reward
    def updateTerminal(self):
        self.num_game += 1
        self.ep_rewards.append(self.ep_reward)
        self.ep_reward = 0.
    def tabulate(self, epsilon, step, agent):
        avg_reward = self.total_reward / self.test_step
        if self.update_count > 0:
            avg_loss = self.total_loss / self.update_count
            avg_q = self.total_q / self.update_count
        else:
            avg_loss = 100
            avg_q = 0

        try:
            max_ep_reward = np.max(self.ep_rewards)
            min_ep_reward = np.min(self.ep_rewards)
            avg_ep_reward = np.mean(self.ep_rewards)
        except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

        entry = np.array([avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, epsilon, self.num_game])
        print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, ep: %.4f, # game: %d' \
          % (entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], entry[6], entry[7]))
        
        if self.stats is None:
            self.stats = entry
        else:
            self.stats = np.vstack((self.stats,entry))

        if step > 180:
            agent.inject_summary({
                'average.reward': avg_reward,
                'average.loss': avg_loss,
                'average.q': avg_q,
                'episode.max reward': max_ep_reward,
                'episode.min reward': min_ep_reward,
                'episode.avg reward': avg_ep_reward,
                'episode.num of game': self.num_game,
                'episode.rewards': self.ep_rewards,
                'episode.actions': self.actions,
                'training.learning_rate': agent.getLR(step),
              }, step)

        self.clear()
        return avg_ep_reward

class Runner(object):
    
    def __init__(self, env, agent, max_step, learn_start, test_step, saveloc):
        self.env = env
        self.agent = agent
        self.max_step = max_step
        self.learn_start = learn_start
        self.test_step = test_step
        self.saveloc = saveloc
        self.statsMgr = Stats(test_step)
        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)
        
    def train(self):
        tf.global_variables_initializer().run()
                
        max_avg_ep_reward = 0
    
        state = self.env.reset()
        
        self.agent.init(state)
        
        qv = self.agent.sampleQS(self.env.get_q_test())
        self.env.showQS(qv)
                    
        start_step = self.step_op.eval()
        
        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step, mininterval=1.5):
            if self.step == self.learn_start:
                self.statsMgr.clear()
    
            # 1. predict
            action = self.agent.predict(self.step)
            # 2. act
            state, reward, terminal, _ = self.env.step(action)
            # 3. observe
            loss, q_t = self.agent.observe(self.step, state, reward, action, terminal)
            
            self.statsMgr.update(loss, q_t, reward, action)
            
            if terminal:
                state = self.env.reset(dorand=True)
                self.statsMgr.updateTerminal()
    
            if self.step >= self.learn_start:
                if self.step % 500 == 0:
                    self.agent.sampleQS(self.env.get_q_test())
                if self.step % self.test_step == self.test_step - 1:
                    avg_ep_reward = self.statsMgr.tabulate(self.agent.getEpsilon(), self.step, self.agent)
                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.step_assign_op.eval({self.step_input: self.step + 1})
                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)
                    
        qv = self.agent.sampleQS(self.env.get_q_test())
        self.env.showQS(qv)
        self.save()
        
    def save(self):
        sv = self.env.getSave()
        qs = self.agent.getQS()
        sdict = {'qs':qs, 'game':sv}
        df = self.saveloc + '/out.pkl'
        if os.path.isfile(df):
            os.remove(df)
        with open(df,'wb') as output:
            pickle.dump(sdict, output)
        np.savetxt(self.saveloc+'/stats.csv', self.statsMgr.stats, delimiter=',')

def main(args):
    with tf.Session() as sess:
        config = get_config(args)
        saveloc = os.getcwd()
        w = TradingGame(config, args.maxgamelen, args.random, saveloc)
        agent = Agent(config, sess)
        runner = Runner(w, agent, config.max_step, config.learn_start, agent.test_step, saveloc)
        
        if args.train:
            runner.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This starts the deepq')
    parser.add_argument('-l', '--load', help='Load from last run',\
                        required=False, dest='load', action='store_true')
    parser.add_argument('-n', '--notrain', help='Don\'t do training', \
                        required=False, dest='train', action='store_false')
    parser.add_argument('-m', '--nomacd', help='No MACD', \
                        required=False, dest='macd', action='store_false')
    parser.add_argument('-r', '--norandom', help='New Random Data', \
                        required=False, dest='random', action='store_false')
    parser.add_argument('--maxgamelen', help='Max length of game', required=False, default=1000, type=int)
    parser.add_argument('--maxgames', help='Max Games', required=False, default=0, type=int)
    parser.add_argument('--model', help='Model to load', required=False, default='trading')    

    args = parser.parse_args()
    main(args)
    