#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:36:20 2019

@author: eadali
"""

from models import mass_spring_damper_model, lstm_model
from numpy import cumsum, zeros, random, float32
from matplotlib import pyplot




# PARAMETERS
# =============================================================================
# Data generation parameters
num_samples = 256 #1024
num_timesteps = 64 #64
split_ratio = 0.2

# Mass-Spring-Damper model parameters
m = 1
k = 8
b = 0.8
x_0 = [0, 0]

# LSTM model parameters
model_shape= [8, 4]
num_lookback = 4
num_epochs = 32
# =============================================================================



# GENERATE DATA FROM MSD MODEL
# =============================================================================
print('Generating data from msd model...')

# Generates input data
x_data = cumsum(random.rand(num_samples,num_timesteps,1)-0.5, axis=1)
y_data = zeros(x_data.shape, dtype=float32)

# Calculates output data with input data and dynamic model
for sample_index, input_signal in enumerate(x_data):
    # Creates dynamic pendulum model
    msd = mass_spring_damper_model(m, k, b, x_0)

    for time_index, u in enumerate(input_signal):
        y_data[sample_index, time_index] = msd.update(u)

# Split training and test data
x_test = x_data[:int(num_samples*split_ratio),]
y_test = y_data[:int(num_samples*split_ratio):,]
x_train = x_data[int(num_samples*split_ratio):,]
y_train = y_data[int(num_samples*split_ratio):,]

print('x_train.shape..:', x_train.shape)
print('y_train.shape..:', y_train.shape)


# Plot one output signal
pyplot.subplot(2,1,1)
pyplot.plot(y_train[0,], 'b')
pyplot.xlabel('Time[s]')
pyplot.ylabel('Position of Mass[m]')
pyplot.grid()

# Plot one input signal
pyplot.subplot(2,1,2)
pyplot.plot(x_train[0,], 'b')
pyplot.xlabel('Time[s]')
pyplot.ylabel('Force[N]')
pyplot.legend(loc='best')
pyplot.grid()

pyplot.tight_layout()
pyplot.show()
# =============================================================================



# TRAIN LSTM MODEL WITH GENERATED MODEL
# =============================================================================
print('Training lstm model...')

num_u = x_train.shape[2]
num_y = y_train.shape[2]

# Creates LSTM model
lstm = lstm_model(model_shape, num_lookback, num_u, num_y)
lstm.fit(x_train, y_train, num_epochs)
# =============================================================================



# TEST LSTM MODEL AND PLOT
# =============================================================================
print('Plotting results...')

# LSTM model predicts mass position
y_pred = zeros(y_test.shape)

for sample_index in range(x_test.shape[0]):
    for time_index in range(x_test.shape[1]):
        y_pred[sample_index, time_index] = lstm.update(x_test[sample_index,time_index,0])

# Plots position of mass
pyplot.subplot(2,1,1)
pyplot.plot(y_test[0,:,0], 'b', label='MSD')
pyplot.plot(y_pred[0,:,0], 'r', label='LSTM')
pyplot.xlabel('Time[s]')
pyplot.ylabel('Position of Mass[m]')
pyplot.legend(loc='best')
pyplot.grid()

# Plots requested torque by PID controller
pyplot.subplot(2,1,2)
pyplot.plot(x_test[0,:,0])
pyplot.xlabel('Time[s]')
pyplot.ylabel('Force[N]')
pyplot.grid()

pyplot.tight_layout()
pyplot.show()
# =============================================================================