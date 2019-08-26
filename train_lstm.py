#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:36:20 2019

@author: eadali
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from models import mass_spring_damper_model
from numpy import cumsum, zeros, random, pi, arange
from matplotlib import pyplot
import numpy




# =============================================================================
# CREATES DATA FROM DYNAMIC MODEL
# =============================================================================

print('Generating data from dynamic pendulum model...')

# Generates input data
x_data = cumsum(random.rand(128,128)-0.5, axis=1)
y_data = zeros(x_data.shape)

# Calculates output data with input data and dynamic model
for sample_index, input_signal in enumerate(x_data):
    # Creates dynamic pendulum model
    msd = mass_spring_damper_model(m=1, k=8, b=0.8, x_0=[0,0])

    for time_index, u in enumerate(input_signal):
        y_data[sample_index, time_index] = msd.update(u)

# Plot one input signal
pyplot.subplot(2,1,1)
pyplot.plot(x_data[20,:], label='input_signal')
pyplot.grid()

# Plot one output signal
pyplot.subplot(2,1,2)
pyplot.plot(y_data[20,:], label='output_signal')
pyplot.grid()

pyplot.show()

xx = numpy.copy(x_data[0,:])
yy = numpy.copy(y_data[0,:])


# =============================================================================
# CREATES LSTM MODEL
# =============================================================================
num_lookback = 4

lstm_model = Sequential()
lstm_model.add(LSTM(32, input_shape=(num_lookback,1), return_sequences=True))
lstm_model.add(LSTM(32))

lstm_model.add(Dense(1))
lstm_model.compile(loss='mse', optimizer=RMSprop())
lstm_model.summary()




# =============================================================================
# TRAINS LSTM MODEL
# =============================================================================
# Number of epochs
epochs = 4
num_lookback = 4


x_data_temp = zeros((x_data.shape[0], x_data.shape[1], num_lookback))

x_data = numpy.pad(x_data, ((0,0),(num_lookback-1,0)),
                            'constant', constant_values=0.0)

for timestep_index in range(x_data_temp.shape[1]):
    x_data_temp[:,timestep_index] = x_data[:,timestep_index:timestep_index+num_lookback]



x_data = x_data_temp.reshape(-1, num_lookback,1)
y_data = y_data.reshape(-1)

# Splits training and validation data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2, random_state=42)


# Splits training and validation data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2, random_state=42)


print(x_data.shape)
print(y_data.shape)
lstm_model.fit(x_train, y_train, epochs=4, verbose=1, validation_data=(x_val, y_val),)
# Saves LSTM model
lstm_model.save('model.h5')

# Predicts model output signal
y_pred = lstm_model.predict(xx)

# Plot Results
#pyplot.subplot(2,1,1)
#pyplot.plot(x_test[0,:])
#pyplot.grid()

pyplot.subplot(2,1,2)
pyplot.plot(yy, 'b')
pyplot.plot(y_pred, 'r')
pyplot.grid()

pyplot.show()