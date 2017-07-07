import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from scipy.stats import norm
import os
import pickle
from tbrn.trading_game import TradingGame

def drawf(qs, glen, gpen, grew):
    plt.ioff()    
    sar = np.empty(shape=(5,3,3,len(qs)))
    for i, items in enumerate(qs):
        for k in range(3):
            for j in range(5):
                sar[j,k,:,i] = items[5*k+j,:]
    
    fnum = 0
    colors=[(0.64,0.04,0.04),'g','w']
    linestyles = ['dotted','solid','-.']
    for j in range(5):
        ax = plt.gca()
        ax.set_axis_bgcolor('black')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        fig = plt.figure(fnum)
        fig.patch.set_facecolor('black')
        plt.title('v{0}'.format(fnum))
        hnd = []
        for i in range(3):
            for k in range(3):
                p, = plt.plot(sar[j,i,k,:],label='{0}-{1}'.format(TradingGame.translatePosition(i),
                                                                  TradingGame.translateAction(k)),
                              color=colors[i],linestyle=linestyles[k],linewidth=3)
                hnd.append(p)
        leg = plt.legend(handles=hnd)
        for text in leg.get_texts():
            text.set_color("white")
        fnum += 1

    ax = plt.gca()
    ax.set_axis_bgcolor('black')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.figure(fnum)
    plt.title('gamelen')
    plt.plot(np.convolve(glen,np.ones((20,))/20,mode='valid'))
    fnum += 1
    plt.figure(fnum)
    plt.title('gamerewpen')
    p, = plt.plot(np.convolve(gpen, np.ones((20,))/20,mode='valid'),label='penalty')
    r, = plt.plot(np.convolve(grew, np.ones((20,))/20,mode='valid'),label='reward')
    plt.legend(handles=[p,r])
    plt.show()

    
if __name__ == '__main__':
    pkl_file = open(os.getcwd() + '/out.pkl', 'rb')
    pdata = pickle.load(pkl_file)
    drawf(pdata['qs'], pdata['game']['glen'], pdata['game']['gpen'], pdata['game']['grew'])
    