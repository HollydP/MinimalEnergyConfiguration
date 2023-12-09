import random
import numpy as np
import math
import matplotlib.pyplot as plt
# import pandas as pd


class Simulated_annealing:
    
    '''
    Handles simulation of queueing system.

    1. Generate charge arrangement
    2. propose new position for 1 charge
    3. determine favorability of new position
    4. If more favourable update,
    '''
    
    def __init__(self,number_charges,max_stepsize,a):
        '''
        Description
        -----------
        Initializes the simulation environment and parameters.
        
        Parameters
        ----------
        '''

        self.n = number_charges
        self.max_stepsize = max_stepsize
        self.a = a

    def charge_arrangement_initial(self):
        # initilize initial charge configuration
        r = np.sqrt(np.random.uniform(0,1,self.n))
        theta = np.random.uniform(0,2*math.pi,self.n)
        x = [r*math.sin(t) for t in theta]
        y = [r*math.cos(t) for t in theta]

        coord = [(i,j) for i in x for j in y]

        # plot charge arrangement
        # r = 1
        # tc = np.linspace(0,2*math.pi,100)
        # xc = [r*math.sin(t) for t in tc]
        # yc = [r*math.cos(t) for t in tc]
        # plt.scatter(x,y)
        # plt.plot(xc,yc)
        # plt.show()

        return coord

    def step(self,current):
        # generate proposed location
        step_r = np.random.random(0,self.max_stepsize)
        theta = np.random.uniform(0,2*math.pi,self.n)

        proposed  = (current[0] + step_r*math.sin(theta),current[1] + step_r*math.cos(theta))
        return proposed
    
    def update(self,temp,current,proposed,prob):
        # determine whether system is updated
        if prob >=1:
            current = proposed
            return current
        elif temp < prob:
            current = proposed
            return
        else:
            return current

    def annealing_func(a,t):
        # annealing function for temperature decay
        return np.exp(-a^t)
    
    def arrangement_energy(self):
        # determine liklihood of move using energy and force
        # calculate energy across total arrangement?
        return 0.5

    def run(self):
        # initilize time
        t=0
        dt = 0.05
        
        # generate initial configuration
        coords = self.charge_arrangement_initial()

        # while annealing temperature is large enough, continue minimising
        while temp > 1e-3:
            # determine temperature as part of annealing process
            temp = self.annealing_func(t)
            t = t+dt

            # select charge to be moved
            charge = np.random.randint(0,self.n)
            x_current = coords[charge]

            # determine potential location
            x_proposed = self.step(x_current)
            likelihood = self.arrangement_energy()
            # update chain
            x_new = self.update(temp,x_current,x_proposed,likelihood)
            coords[charge] = x_new
            


        



