import numpy as np
import pickle
import os

class Game(object):
    def __init__(self):
        pass
    def reset(self):
        # reset logic
        return self._get_state()
    def step(self, action):
        # step logic
        return self._get_state(), self._get_reward(), self._is_game_over(), None
    def render(self):
        print self._get_state(), self._get_action()
    def get_q_test(self):
        raise NotImplementedError('get_q_test not implemented')
    def showQS(self, qs):
        pass
    def getName(self):
        return self.name
    def _get_state(self):
        return self.state
    def _get_reward(self):
        return self.reward
    def _is_game_over(self):
        if (self.current_step == self.n_steps) | self.exited:
            return True
        else:
            return False

class QHandler(object):
    def __init__(self, init_data, num_st, input_size, output_size):
        self.num_st = num_st
        self.input_size = input_size
        self.output_size = output_size
        self._prepQS(init_data)

    def _prepQS(self, init_data):
        mlen = self.output_size * self.num_st
        entries = np.empty(shape=(mlen,self.input_size))
        pos = 0
        for j in range(self.output_size):
            for x in range(self.num_st):
                entries[pos] = [init_data[x],j]
                pos += 1
        self.testSet = entries
        
    @staticmethod
    def showQSDetail(ds, translator):
        for i in range(ds.shape[0]):
            print('{0} : {1}'.format(str(ds[i,]), translator(np.argmax(ds[i,]))))

    def showQS(self, qs, translator):
        print('for short:')
        QHandler.showQSDetail(qs[0:self.num_st], translator)
        print('for long:')
        QHandler.showQSDetail(qs[self.num_st:self.num_st*2], translator)
        print('for flat:')
        QHandler.showQSDetail(qs[self.num_st*2:self.num_st*3], translator)

class DataSource(object):
    def __init__(self, use_random, saveloc, nprod, maxlen, num_st, amp):
        self.maxlen = maxlen
        self.num_st = num_st
        if use_random:
            self.x = np.empty(shape=(nprod, maxlen))
            self._generate_triangle(amp)
            df = saveloc + '/data.pkl'
            if os.path.isfile(df):
                os.remove(df)
            with open(saveloc + '/data.pkl','wb') as output:
                pickle.dump({'x': self.x}, output)
        else:
            pkl_file = open(saveloc + '/data.pkl', 'rb')
            pdata = pickle.load(pkl_file)
            self.x = pdata['x']
            
    def reset(self, dorand=False):
        if dorand:
            self.clock = np.random.randint(0,(self.num_st - 1) * 2)
        else:
            self.clock = 0

    def _generate_triangle(self, amp):
        div = self.num_st - 1
        yinc = amp / float(div)
        self.x[0,:] = [abs(div*((i/div)%2)-(i%div))*yinc - 0.5*amp for i in range(self.maxlen)]
        self.x[1,:] = -self.x[0,:]
        
    def get_val(self, offset=0):
        if offset + self.clock > self.maxlen-1:
            return self.x[0, self.maxlen-1]
        return self.x[0, self.clock + offset]

class TradingGame(Game):
    
    S_FLAT = 2
    S_LONG = 1
    S_SHORT = 0
    A_NOTHING = 2
    A_BUY = 1
    A_SELL = 0
    
    def __init__(self, config, maxlen, use_random, saveloc, nprod=2):
        
        self.num_st = config.num_st
        self.amp = int((config.num_st - 1) * config.amp_scale)
        self.same_penalty = config.samepenalty
        self.reward_scale = config.rewardscale
        
        self.name = 'TradingGame'
        self.maxlen = maxlen
        
        self.state = np.empty(config.input_size)
        
        self.data = DataSource(use_random, saveloc, nprod, maxlen, self.num_st, self.amp)
        self.qhandler = QHandler(self.data.x[0], self.num_st, config.input_size, config.action_size)

        self._initStats()
        self.reset()
        float_formatter = lambda x: "%.3f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})
        
    def _initStats(self):
        self.glen = []
        self.gpen = []
        self.grew = []
        self.costs = []
        self.rewards = []
        self.actionsmem = []
        self.numChange = 0
        
    def get_q_test(self):
        return self.qhandler.testSet

    def reset(self, dorand=False):
        self.data.reset(dorand)
        self.position = TradingGame.S_FLAT
        self.last = None
        self.cumpenalty = 0
        self._fill_state()
        return self._get_state()
    
    def _get_state(self):
        return np.copy(self.state)
    
    def render(self):
        print self._get_state()
        
    def _fill_state(self):
        self.state[0] = self.data.get_val()
        self.state[1] = self.position

    def _getUpdate(self, action):
        #ret state, penalty, reward, done
        if action == TradingGame.A_NOTHING:
            return self.position, 0, 0, False
        v1 = self.data.get_val(1)
        v0 = self.data.get_val()
        if self.position == TradingGame.S_FLAT:
            if action == TradingGame.A_BUY:
                return TradingGame.S_LONG, 0, self.reward_scale*(v1 - v0), False
            if action == TradingGame.A_SELL:
                return TradingGame.S_SHORT, 0, self.reward_scale*(v0 - v1), False
        if self.position == TradingGame.S_SHORT:
            if action == TradingGame.A_BUY:
                return TradingGame.S_FLAT, 0, self.last[1] - v0, True
            if action == TradingGame.A_SELL:
                return TradingGame.S_SHORT, self.same_penalty, 0, False
        if self.position == TradingGame.S_LONG:
            if action == TradingGame.A_BUY:
                return TradingGame.S_LONG, self.same_penalty, 0, False
            if action == TradingGame.A_SELL:
                return TradingGame.S_FLAT, 0, v0 - self.last[1], True
    
    def step(self, action):
        
        penalty = 0.0
        reward = 0
        position = self.position
        done = False
        if action != TradingGame.A_NOTHING:
            position, penalty, reward, done = self._getUpdate(action)
            self.cumpenalty += penalty
            if not self.last:
                self.last = (self.data.clock, self.data.get_val())
            if done:
                self.glen.append(self.data.clock - self.last[0])
                self.gpen.append(self.cumpenalty)
                self.grew.append(reward)
            self.actionsmem.append([self.data.clock, action])
            self.numChange += 1
        
        self.position = position
        self.data.clock += 1
        self._fill_state()
        
        if self.data.clock == self.maxlen:
            penalty += self.same_penalty * 50.0 # Very bad
            done = True
        
        return self._get_state(), (reward - penalty), done, None
    
    def getSave(self):
        return {'glen':self.glen,'gpen':self.gpen,'grew':self.grew}
                
    @staticmethod
    def translatePosition(position):
        if position == TradingGame.S_FLAT:
            return 'FLAT'
        elif position == TradingGame.S_LONG:
            return 'LONG'
        else:
            return 'SHORT'
    
    @staticmethod
    def translateAction(action):
        if action == TradingGame.A_NOTHING:
            return 'NOTHING'
        elif action == TradingGame.A_BUY:
            return 'BUY'
        elif action == TradingGame.A_SELL:
            return 'SELL'

    def showQS(self, qs):
        self.qhandler.showQS(qs, TradingGame.translateAction)
