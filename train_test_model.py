#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:36:20 2019

@author: eadali
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from models import pendulum_model
from numpy import cumsum, zeros, random, pi, arange
from matplotlib import pyplot




# =============================================================================
# CREATES TRAINING DATA FROM DYNAMIC MODEL
# =============================================================================

print('Generating data from dynamic pendulum model...')

# Generates input data
x_data = cumsum(random.rand(128,128)-0.5, axis=1)
y_data = zeros(x_data.shape)

# Calculates output data with input data and dynamic model
for sample_index, input_signal in enumerate(x_data):
    # Creates dynamic pendulum model
    pendulum = pendulum_model(0.25, 5, [0,0])

    for time_index, u in enumerate(input_signal):
        y_data[sample_index, time_index] = pendulum.update(u)

    print(sample_index)
    if sample_index > 10:
        break

## TODO: split is not working correctly
indeces = arange(x_data.shape[0])
random.shuffle(indeces)
split_index = int(0.32 * indeces.shape[0])


x_train = x_data[split_index:]
y_train = y_data[split_index:]
x_test = x_data[:split_index]
y_test = y_data[:split_index]

#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
#x_train, y_train = x_data, y_data

pyplot.subplot(2,1,1)
pyplot.plot(x_train[0,:], label='input_signal')
pyplot.grid()

pyplot.subplot(2,1,2)
pyplot.plot(y_train[0,:], label='output_signal')
pyplot.grid()

pyplot.show()




# =============================================================================
# CREATES LSTM MODEL
# =============================================================================
model = Sequential()
model.add(LSTM(8, batch_input_shape=(1,1,1), stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()




# =============================================================================
# TRAINS LSTM MODEL
# =============================================================================
epochs = 64

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

for epoch_index in range(epochs):
    print('epoch', epoch_index + 1, '/', epochs)

    for sample_index in range(x_train.shape[0]):
        model.fit(x_train[sample_index,:].reshape(128,1,1), y_train[sample_index,:], batch_size=1, epochs=1,
                  verbose=1,  shuffle=False)
        model.reset_states()


#validation_data=(x_val, y_val),

#
#y_pred = model.predict(x_test, batch_size=1)
#
#pyplot.subplot(2,1,1)
#pyplot.plot(x_test[0,:])
#pyplot.plot(2,1,2)
#pyplot.plot(y_test[0,:])
#pyplot.plot(y_pred[0,:])
#pyplot.show()
#
#
