#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:36:20 2019

@author: eadali
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from models import mass_spring_damper_model
from numpy import cumsum, zeros, random, pad
from matplotlib import pyplot






# =============================================================================
# GENERATES DATA FROM DYNAMIC MODEL
# =============================================================================
# Data generation parameters
split_ratio = 0.2

num_samples = 128
num_timesteps = 128
num_lookback = 4


# Mass-Spring-Damper model parameters
m = 1
k = 8
b = 0.8
x_0 = [0, 0]

print('Generating data from dynamic pendulum model...')

# Generates input data
x_data = cumsum(random.rand(num_samples,num_timesteps)-0.5, axis=1)
y_data = zeros(x_data.shape)

# Calculates output data with input data and dynamic model
for sample_index, input_signal in enumerate(x_data):
    # Creates dynamic pendulum model
    msd = mass_spring_damper_model(m, k, b, x_0)

    for time_index, u in enumerate(input_signal):
        y_data[sample_index, time_index] = msd.update(u)

# Reshape data for LSTM model
x_data_temp = zeros((x_data.shape[0], x_data.shape[1], num_lookback, 2))

x_data = pad(x_data, ((0,0),(num_lookback-1,0)),
                            'constant', constant_values=0.0)

y_data = pad(y_data, ((0,0),(num_lookback,0)),
                            'constant', constant_values=0.0)


for time_index in range(x_data_temp.shape[1]):
    x_data_temp[:,time_index,:,0] = x_data[:,time_index:time_index+num_lookback]
    x_data_temp[:,time_index,:,1] = y_data[:,time_index:time_index+num_lookback]


x_data = x_data_temp
y_data = y_data[:,num_lookback:]

#x_data = x_data_temp.reshape(-1, num_lookback, num_features)
#y_data = y_data[:,num_lookback:].reshape(-1)


# Split data to training, validation and test
test_index = int(num_samples * split_ratio)
val_index = test_index * 2

x_test = x_data[:test_index]
y_test = y_data[:test_index]
x_val = x_data[test_index:val_index]
y_val = y_data[test_index:val_index]
x_train = x_data[val_index:]
y_train = y_data[val_index:]


print('x_train.shape..:', x_train.shape)
print('y_train.shape..:', y_train.shape)
print('x_val.shape..:', x_val.shape)
print('y_val.shape..:', y_val.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)


# Plot one input signal
pyplot.subplot(2,1,1)
pyplot.plot(x_test[0,:,3,0], '.-b', label='input_signal')
pyplot.grid()

# Plot one output signal
pyplot.subplot(2,1,2)
pyplot.plot(x_test[0,:,3,1], '.-b', label='input_signal')
pyplot.plot(y_test[0,:], '.-r', label='output_signal')
pyplot.grid()

pyplot.show()




# =============================================================================
# CREATES LSTM MODEL
# =============================================================================
num_timesteps = x_train.shape[2]
num_lookback = x_train.shape[3]

# Creates LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(8, input_shape=(num_timesteps,num_lookback), return_sequences=True))
lstm_model.add(LSTM(4))

lstm_model.add(Dense(1))
lstm_model.compile(loss='mse', optimizer=Adam())

lstm_model.summary()




# =============================================================================
# TRAINS LSTM MODEL
# =============================================================================
num_lookback = x_train.shape[2]
num_features = x_train.shape[3]

# Reshape training data for LSTM model
x_train = x_train.reshape(-1, num_lookback, num_features)
y_train = y_train.reshape(-1)
x_val = x_val.reshape(-1, num_lookback, num_features)
y_val = y_val.reshape(-1)

print(x_train.shape)
print(y_train.shape)

# Trains LSTM model
checkpoint = ModelCheckpoint('lstm_model.h5', save_best_only=True)
lstm_model.fit(x_train, y_train, epochs=256,
               verbose=1, validation_data=(x_val, y_val),
               callbacks=[checkpoint,])


# Saves LSTM model
lstm_model.save('model.h5')


#TODO: Implement Evaluation
# Predicts model output signal
y_pred = zeros(y_test.shape[1])
for time_index, x in enumerate(x_test[0,]):
    y_pred[time_index] =

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