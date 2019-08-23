#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:36:20 2019

@author: eadali
"""

from scipy.integrate import odeint
import numpy
from matplotlib import pyplot




class pendulum:
    def __init__(self):
        ''' Inits pendulum constants and initial state
        '''

        self.constant_1 = 0.25
        self.constant_2 = 5.0
        self.initial_state = [0, 0]



    def pendulum(self, states, ttime, force):
        ''' Dynamic equations of pendulum
        states: [theta, omega]
        ttime: time steps for ode solving
        forse: external force applied to the pendulum
        '''

        theta, omega = states
        states_dt = [omega, -self.constant_1*omega - self.constant_2*numpy.sin(theta)+force]

        return states_dt



    def update(self, force):
        ''' Interface function for pendulum model
        force: External force applied to the pendulum
        '''
        states = odeint(self.pendulum, self.initial_state, [0,0.1], args=(force,))

        self.initial_state = states[1]

        return states[1]




if __name__ == '__main__':
    ''' Test of pendulum class
    '''

    pendulum_model = pendulum()
    theta = list()

    for time_step in range(512):
        theta.append(pendulum_model.update(0.4)[0])

    pyplot.plot(theta, label='theta(t)')
    pyplot.legend(loc='best')
    pyplot.xlabel('t')
    pyplot.grid()
    pyplot.show()
