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

# Splits training and test data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Plot one input signal
pyplot.subplot(2,1,1)
pyplot.plot(x_train[20,:], label='input_signal')
pyplot.grid()

# Plot one output signal
pyplot.subplot(2,1,2)
pyplot.plot(y_train[20,:], label='output_signal')
pyplot.grid()

pyplot.show()




# =============================================================================
# CREATES LSTM MODEL
# =============================================================================
model = Sequential()
model.add(LSTM(32, input_shape=(1,1), batch_size=1, return_sequences=True, stateful=True))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer=RMSprop())
model.summary()




# =============================================================================
# TRAINS LSTM MODEL
# =============================================================================
# Number of epochs
epochs = 256

# Splits training and validation data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2, random_state=42)
# Epoch cycle
for epoch_index in range(epochs):
    print('epoch', epoch_index + 1, '/', epochs)

    # Sample cycle
    training_loss = 0
    for sample_index in range(x_train.shape[0]):
        x_sample = x_train[sample_index,:].reshape(-1,1,1)
        y_sample = y_train[sample_index,:]

        # Trains model
        history = model.fit(x_sample, y_sample,
                            batch_size=1, epochs=1, verbose=0,  shuffle=False)

        training_loss = training_loss + history.history['loss'][0]

        # Resets model after sample fit
        model.reset_states()

    # Evaluate model after each epoch
    validation_loss = 0
    for sample_index in range(x_val.shape[0]):
        x_sample = x_val[sample_index,:].reshape(-1,1,1)
        y_sample = y_val[sample_index,:]

        # Calculates loss for each sample
        validation_loss = validation_loss + model.evaluate(x_sample, y_sample, batch_size=1, verbose=0)

        # Resets model after sample evaluate
        model.reset_states()

    print('average training loss..:', training_loss / x_train.shape[1])
    print('average validation loss..:', validation_loss / x_val.shape[1])

# Saves LSTM model
model.save('model.h5')

# Predicts model output signal
y_pred = model.predict(x_test[0,:].reshape(-1,1,1), batch_size=1)

# Plot Results
pyplot.subplot(2,1,1)
pyplot.plot(x_test[0,:])
pyplot.grid()

pyplot.subplot(2,1,2)
pyplot.plot(y_test[0,:], 'b')
pyplot.plot(y_pred, 'r')
pyplot.grid()

pyplot.show()