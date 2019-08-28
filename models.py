#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:36:20 2019

@author: eadali
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


from scipy.integrate import odeint
from numpy import sin, copy, zeros, float32, pad

from matplotlib import pyplot




class pendulum_model:
    def __init__(self, m, l, b, g, x_0):
        """Inits pendulum constants and initial state

        # Arguments
            m: Pendulum mass
            l: Pendulum length
            b: Pendulum friction coeff
            g: Earth gravity acceleration
            x_0: Pendulum initial state
        """

        self.m = m
        self.l = l
        self.b = b
        self.g = g

        self.x_0 = x_0



    def ode(self, x, t, u):
        """Dynamic equations of pendulum

        # Arguments
            x: [angle of pendulum, angular velocity of pendulum]
            t: Time steps for ode solving
            u: External force applied to the pendulum

        # Returns
            Derivative of internal states
        """

        # Calculates equation coeffs
        c_1 = -self.b/(self.m*self.l**2)
        c_2 = -self.g/self.l
        c_3 = 1.0/(self.m*self.l**2)

        # ODE of pendulum
        theta, omega = x
        dxdt = [omega, c_1*omega + c_2*sin(theta) + c_3*u]

        return dxdt



    def update(self, u):
        """Interface function for pendulum model

        # Arguments
            u: External force applied to the pendulum

        # Returns
            Angle of pendulum
        """

        # Solving ODE with scipy library
        x = odeint(self.ode, self.x_0, [0,0.1], args=(u,))

        self.x_0 = x[1]

        return x[1,0]




class mass_spring_damper_model:
    def __init__(self, m, k, b, x_0):
        """Inits pendulum constants and initial state

        # Arguments
            m: Mass
            k: Spring coeff
            b: Friction coeff
            x_0: Initial state
        """

        self.m = m
        self.k = k
        self.b = b
        self.x_0 = x_0



    def ode(self, x, t, u):
        """Dynamic equations of pendulum

        # Arguments
            x: [position of mass, velocity of mass]
            t: Time steps for ode solving
            u: External force applied to the mass

        # Returns
            Derivative of internal states
        """

        # ODE of mass-spring-damper model
        pos, acc = x
        dxdt = [acc, -(self.b/self.m)*acc - (self.k/self.m)*pos + (1/self.m)*u]

        return dxdt



    def update(self, u):
        """Interface function for pendulum model

        # Arguments
            u: External force applied to the mass

        # Returns
            Position of mass
        """

        # Solving ODE with scipy library
        x = odeint(self.ode, self.x_0, [0,0.1], args=(u,))

        self.x_0 = x[1]

        return x[1,0]




class lstm_model:
    def __init__(self, model_shape, num_lookback, num_u, num_y):
        self.x = zeros((1,num_lookback,num_u+num_y,))
        self.model = Sequential()

        num_layers = len(model_shape)

        if self.equal(num_layers, 1):
            num_cells = model_shape[0]
            self.model.add(LSTM(num_cells, input_shape=(num_lookback,num_u+num_y)))

        else:
            num_cells = model_shape[0]
            self.model.add(LSTM(num_cells, input_shape=(num_lookback,num_u+num_y),
                                return_sequences=True))

            for num_cells in model_shape[1:-1]:
                self.model.add(LSTM(num_cells, return_sequences=True))

            num_cells = model_shape[-1]
            self.model.add(LSTM(num_cells))

        self.model.add(Dense(num_y))
        self.model.compile(loss='mse', optimizer='adam')

        self.model.summary()



    def equal(self, val_1, val_2):
        condition_1 = (val_1 > (val_2-0.0001))
        condition_2 = (val_1 < (val_2+0.0001))

        return condition_1 and condition_2



    def _reshape(self, x_data, y_data):
        x_data = copy(x_data)
        y_data = copy(y_data)



        _, num_lookback, num_x = self.model.layers[0].input_shape
        _, num_y = self.model.layers[-1].output_shape

        new_shape = (x_data.shape[0], x_data.shape[1]-num_lookback,
                     num_lookback, num_x)
        x_data_new = zeros(new_shape, dtype=float32)

        x_data = x_data[:,1:,]

        for time_index in range(x_data_new.shape[1]):
            x_data_new[:,time_index,:,0:num_x-num_y] = x_data[:,time_index:time_index+num_lookback]
            x_data_new[:,time_index,:,num_x-num_y:num_x] = y_data[:,time_index:time_index+num_lookback]


        y_data_new = y_data[:,num_lookback:,]

        pyplot.subplot(2,1,1)
        pyplot.plot(x_data_new[0,:,3,0], '.-b', label='input_signal')
        pyplot.grid()

        # Plot one output signal
        pyplot.subplot(2,1,2)
        pyplot.plot(x_data_new[0,:,3,1], '.-b', label='input_signal')
        pyplot.plot(y_data_new[0,:], '.-r', label='output_signal')
        pyplot.grid()

        pyplot.show()

        print(x_data_new.shape)
        print(y_data_new.shape)
        return x_data_new, y_data_new



    def fit(self, x_data, y_data, validation_split=0.2, path='temp_model.h5'):
        x_data = copy(x_data)
        y_data = copy(y_data)

        x_data, y_data = self._reshape(x_data, y_data)

        _, num_lookback, num_x = self.model.layers[0].input_shape
        _, num_y = self.model.layers[-1].output_shape

        x_data = x_data.reshape(-1, num_lookback, num_x)
        y_data = y_data.reshape(-1)


        # Trains LSTM model
        checkpoint = ModelCheckpoint(path, save_best_only=True)
        self.model.fit(x_data, y_data, epochs=256,
                       verbose=1, validation_split=validation_split,
                       callbacks=[checkpoint,])

        self.lstm_model = load_model(path)



    def load(self, path):
        self.lstm_model = load_model(path)



    def update(self, u):
        self.x[0,-1,0:] = u
        y_pred = self.model.predict(self.x)

        self.x[0,-1,1:] = y_pred

        return y_pred[0]







if __name__ == '__main__':
    """Test of models
    """

    # Test of pendulum_model class
    pendulum = pendulum_model(m=1, l=1, b=0.25, g=9.8, x_0=[1,0])
    theta = list()

    for t in range(512):
        theta.append(pendulum.update(8))

    pyplot.plot(theta, label='theta(t)')
    pyplot.legend(loc='best')
    pyplot.xlabel('t')
    pyplot.grid()
    pyplot.show()

    # Test of mass_spring_damper_model class
    msd = mass_spring_damper_model(m=1, k=8, b=0.8, x_0=[1,0])
    pos = list()

    for t in range(512):
        pos.append(msd.update(0.4))

    pyplot.plot(pos, label='pos(t)')
    pyplot.legend(loc='best')
    pyplot.xlabel('t')
    pyplot.grid()
    pyplot.show()


