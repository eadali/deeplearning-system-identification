#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:36:20 2019

@author: eadali
"""

from scipy.integrate import odeint
import numpy
from matplotlib import pyplot




class pendulum_model:
    def __init__(self, b, c, x_0):
        """Inits pendulum constants and initial state

        # Arguments
            b: Pendulum constant 1
            c: Pendulum constant 2
            x_0: Initial state
        """

        self.b = b
        self.c = c
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

        # ODE of pendulum
        theta, omega = x
        dxdt = [omega, -self.b*omega - self.c*numpy.sin(theta)+u]

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
        self.m = m
        self.k = k
        self.b = b
        self.x_0 = x_0

    def ode(self, x, t, u):
        pos, acc = x

        dxdt = [acc, -(self.b/self.m)*acc - (self.k/self.m)*pos + (1/self.m)*u]

        return dxdt

    def update(self, u):
        # Solving ODE with scipy library
        x = odeint(self.ode, self.x_0, [0,0.1], args=(u,))

        self.x_0 = x[1]

        return x[1,0]





if __name__ == '__main__':
    """Test of models
    """

    # Test of pendulum_model class
#    pendulum = pendulum_model(b=0.25, c=5, x_0=[1,0])
#    theta = list()
#
#    for t in range(512):
#        theta.append(pendulum.update(0.4))
#
#    pyplot.plot(theta, label='theta(t)')
#    pyplot.legend(loc='best')
#    pyplot.xlabel('t')
#    pyplot.grid()
#    pyplot.show()

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


