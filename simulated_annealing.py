import random
import numpy as np
import math
import matplotlib.pyplot as plt
# import pandas as pd


class SimulatedAnnealing:
    
    '''
    Simulated annealing process to minimise charge arrangement

    1. Generate charge arrangement - charge_arrangement_initial
    2. propose new position for 1 charge - step
    3. determine favorability of new position - arrangement_energy_func (needs serious work)
    4. If more favourable update else determine likelihood using annealing temp - annealing_func,update
    %. repeat while annealing temp large...
    '''
    
    def __init__(self,number_charges,max_stepsize,decay_speed,dt):
        '''
        Description
        -----------
        Initializes the simulation environment and parameters.
        
        Parameters
        ----------
        '''

        self.n = number_charges
        self.max_stepsize = max_stepsize
        self.decay_speed = decay_speed
        self.dt = dt

    def charge_arrangement_initial(self):
        # initilize initial charge configuration
        r = np.sqrt(np.random.uniform(0,1,self.n))
        theta = np.random.uniform(0,2*math.pi,self.n)
        x = [r[i]*math.sin(theta[i]) for i in range(self.n)]
        y = [r[i]*math.cos(theta[i]) for i in range(self.n)]
        
        coord = [[x[i],y[i]] for i in range(self.n)]

        # # plot charge arrangement
        # r = 1
        # tc = np.linspace(0,2*math.pi,100)
        # xc = [r*math.sin(t) for t in tc]
        # yc = [r*math.cos(t) for t in tc]
        # plt.scatter(x,y)
        # plt.plot(xc,yc)
        # plt.show()

        return coord
    
        
    def annealing_func(self,t):
        # annealing function for temperature decay
        return np.exp(-self.decay_speed*t)

    def step(self,current):
        # generate proposed location
        proposed = [2,2]
        while sum(np.array(proposed)**2) > 1:
            step_r = np.random.uniform(0,self.max_stepsize)
            theta = np.random.uniform(0,2*math.pi)

            proposed  = [current[0] + step_r*math.sin(theta),current[1] + step_r*math.cos(theta)]

        return proposed
    
    def arrangement_energy(self,coords,x_current,x_proposed):
        # determine liklihood of move using energy and force
        # calculate energy across total arrangement?

        x_ij = np.array([(np.array(x_current[0]) - np.array(coords[j][0])) for j in range(self.n) if coords[j] not in [x_current]])
        y_ij = np.array([(np.array(x_current[1]) - np.array(coords[j][1])) for j in range(self.n) if coords[j] not in [x_current]])
        abs_r = np.sqrt((x_ij**2+y_ij**2))
        current_force = 1/sum(abs_r**2)

        x_ij = np.array([(np.array(x_proposed[0]) - np.array(coords[j][0])) for j in range(self.n) if coords[j] not in [x_current]])
        y_ij = np.array([(np.array(x_proposed[1]) - np.array(coords[j][1])) for j in range(self.n) if coords[j] not in [x_current]])
        abs_r = np.sqrt((x_ij**2+y_ij**2))
        abs_r = np.sqrt((x_ij**2+y_ij**2))
        proposed_force= 1/sum(abs_r**2)

        # positive force is repulsive
        # smaller force is more favourable so should be curr/prop
        likelihood = current_force/proposed_force
        # print(likelihood)

        return likelihood
    
    def update(self,temp,current,proposed,likelihood):
        # determine whether system is updated
        if likelihood >=1:
            current = proposed
            return current
        elif temp > likelihood:
            current = proposed
            return current
        else:
            return current
        
    def calc_energy(self,coords):
        individual_forces =[]
        for i in range(self.n):
            x = coords[i]
            x_ij = np.array([(np.array(x[0]) - np.array(coords[j][0])) for j in range(self.n) if coords[j] not in [x]])
            y_ij = np.array([(np.array(x[1]) - np.array(coords[j][1])) for j in range(self.n) if coords[j] not in [x]])
            abs_r = np.sqrt((x_ij**2+y_ij**2))
            force = 1/sum(abs_r**2)
            individual_forces.append(force)
        energy = sum(individual_forces)

        return energy


    def run(self):
        # initilize time
        t=0
        # initilize temp
        temp = 1
        
        # generate initial configuration
        initial_coords = self.charge_arrangement_initial()
        coords = initial_coords.copy()

        # while annealing temperature is large enough, continue minimising
        while temp > 1e-3:
            # determine temperature as part of annealing process
            temp = self.annealing_func(t)
            t += self.dt

            # select charge to be moved
            charge = np.random.randint(0,self.n)
            x_current = coords[charge]

            # determine potential location
            x_proposed = self.step(x_current)
            likelihood = self.arrangement_energy(coords,x_current,x_proposed)
            # update chain
            x_new = self.update(temp,x_current,x_proposed,likelihood)
            coords[charge] = x_new
            energy = self.calc_energy(coords)
            print("current energy of configuration ",energy)
            print('annealing time ',t)

        
        plt.scatter([x[0] for x in initial_coords],[x[1] for x in initial_coords],c='r')
        plt.scatter([x[0] for x in coords],[x[1] for x in coords],c='b')
        r = 1
        tc = np.linspace(0,2*math.pi,100)
        xc = [r*math.sin(t) for t in tc]
        yc = [r*math.cos(t) for t in tc]
        plt.plot(xc,yc)
        plt.show()
        return



        



