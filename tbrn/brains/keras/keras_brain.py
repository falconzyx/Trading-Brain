"""
In this example we demonstrate how to implement a DQN agent and
train it to trade optimally on a periodic price signal.
Training time is short and results are unstable.
Do not hesitate to run several times and/or tweak parameters to get better results.
Inspired from https://github.com/keon/deep-q-learning
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tbrn.base.brain import Brain

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
        
    def train(self, X, y, batch_size, epochs, verbose):
        vals = self.brain.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)
        return vals.history['loss'][0]
