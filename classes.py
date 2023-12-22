import copy
import math 
import random
import sys 

import matplotlib.pyplot as plt 
import numpy as np


class ChargeCollection:

    '''
    This class represents a collection of charges.
    '''

    def __init__(self, N) -> None:
        self.N = N
        self.charges = self.initialize_charge_arrangement()


    def initialize_charge_arrangement(self):
        '''
        Initializes random configuration of charges.
        '''
        r = np.sqrt(np.random.uniform(0, 1, self.N))
        theta = np.random.uniform(0, 2 * math.pi, self.N)

        x = r * np.cos(theta) 
        y = r * np.sin(theta)
        
        return [list(coords) for coords in zip(x, y)]
    
    
    def get_total_energy_new(self):
        '''
        Returns the total energy of the current charge configuration.
        '''
        # loop over all charge pairs
        # TODO: iterations over all charges could be done more efficiently using numpy
        E = 0
        for i in range(len(self.charges)):
            for j in range(i+1, len(self.charges)):
                r = math.dist(self.charges[i], self.charges[j])
                E += 1/r

        # if particle is out of bounds sure it is not accepted
        # TODO: this doesn't seem correct
        if not self.is_solution():
            return sys.maxsize
        
        return E  # because each contribution was counted twice
    

    def get_total_energy(self):
        """
        Returns the total energy of the charge configuration.
        """
        # loop over all charge pairs
        E = 0
        for charge in self.charges:
            for other_charge in self.charges:
                r = math.dist(charge, other_charge) 
                E += 1/r if r != 0 else 0
        # if particle out of bounds sure it is not accepted
        if not self.is_solution():
            return sys.maxsize
        return E / 2 # because each contribution was counted twice


    def plot_charges(self):
        """
        Plots current charge configuration.
        """
        x_coords = [coord[0] for coord in self.charges]
        y_coords = [coord[1] for coord in self.charges]

        theta = np.linspace(0, 2*np.pi, 100)
        x, y = np.cos(theta), np.sin(theta)

        plt.figure(figsize=(5, 5))
        plt.plot(x, y, label='boundary')
        plt.plot(x_coords, y_coords, "ko")
        plt.axis('equal')
        plt.show()
    

    def move_charge(self, charge, x_step, y_step):
        '''
        Description
        -----------
        Moves a charge with a given step size.

        Parameters
        ----------
        charge : `int`
            The index of the charge to be moved.
        x_step : `float`
            The step size in the x direction.
        y_step : `float`
            The step size in the y direction.
        '''
        new_x = self.charges[charge][0] + x_step
        new_y = self.charges[charge][1] + y_step 

        # Place on circle if step pushes charge out of bounds TODO: improve this
        while np.sqrt(new_x**2 + new_y**2) > 1:
            new_x = new_x - 0.005 if new_x > 0 else new_x + 0.005
            new_y = new_y - 0.005 if new_y > 0 else new_y + 0.005
            
        self.charges[charge][0], self.charges[charge][1] = new_x, new_y


    def is_solution(self):
        '''
        Checks if the charge configuration is valid.
        '''
        for coord in self.charges:
            if math.dist([0, 0], coord) > 1:
                return False
        return True


class SimulatedAnnealing():

    '''
    Implementation of simulated annealing algorithm.
    
    Depends on ChargeCollection class.
    
    Per iteration temperature decreases and the charge configuration is changed randomly,
    - If system energy decreased, accept.
    - If system energy increased, accept or reject given a probability.
    
    To decrease the rate of accepting increased energy, the probability 
    depends on the current temperature.
    '''
    
    def __init__(self, charges:ChargeCollection, max_stepsize, cooling_rate=1e-3,chain_length = 10, init_temperature=5000):
        '''
        Description
        -----------
        Initializes the simulated annealing algorithm.

        Use the `run` method to run the algorithm.

        Parameters
        ----------
        charges : `ChargeCollection`
            The collection of charges to be used.
        max_stepsize : `float`
            The maximum step size for the charges.
        cooling_rate : `float`
            The cooling rate for the temperature.
        init_temperature : `float`
            The initial temperature.
        '''
        if not charges.is_solution():
            raise Exception("HillClimber requires a valid configuration")
        
        # Initialize parameters
        self.charges = copy.deepcopy(charges)
        self.max_stepsize = max_stepsize
        self.cooling_rate = cooling_rate
        self.chain_length = chain_length

        self.energy = charges.get_total_energy()
        self.energy_arr = None 
        
        self.T_arr = None
        self.T0 = init_temperature
        self.T = init_temperature


    def rank_charges(self, neighbor):
        '''
        Ranks the charges per neighboring procedure.

        Parameters
        ----------
        'random' : returns random order of charges
        'iterative' : returns charges in order of index
        'by_distance' : returns charges in order of distance to origin
        'by_force' : returns charges in order of force on charge
        '''
        if neighbor == 'random':
            charge_indices = np.random.choice(self.charges.N, size=self.charges.N, replace=False)
        
        elif neighbor == 'iterative':
            charge_indices = range(self.charges.N)
        
        elif neighbor == 'by_distance':
            charge_indices = np.argsort([math.dist([0, 0], coord) for coord in self.charges.charges])
            charge_indices = charge_indices[::-1] # reverse order
        
        elif neighbor == 'by_force':
            # check force on each charge
            tot_F_ij = np.zeros((self.charges.N, 2))
            norms_F_ij = np.zeros(self.charges.N)
            for i in range(self.charges.N):
                for j in range(self.charges.N):
                    if i != j:
                        r_ij = np.array(self.charges.charges[i]) - np.array(self.charges.charges[j])
                        F_ij = r_ij / np.linalg.norm(r_ij)**3
                        tot_F_ij[i] += F_ij
                norms_F_ij[i] = np.linalg.norm(tot_F_ij[i])
            charge_indices = np.argsort(norms_F_ij)
        
        else:
            raise Exception("Invalid neighbor selection method")
        
        return charge_indices

    def move_charge(self, new_configuration:ChargeCollection, charge_index):
        '''
        Moves a charge at random.

        Parameters
        ----------
        new_configuration : `ChargeCollection`
            The new charge configuration.
        charge_index : `int`
            The index of the charge to be changed.
        '''
        x_step = random.uniform(-self.max_stepsize, self.max_stepsize) 
        y_step = random.uniform(-self.max_stepsize, self.max_stepsize)
        new_configuration.move_charge(charge_index, x_step, y_step)


    def update_temperature(self,linear=False):
        '''
        Description
        -----------
        Handles cooling procedure.

        Parameters
        ----------
        linear : `bool`
            Whether to use a linear cooling procedure instead of exponential.
        '''
        if not linear:
            self.T = self.T * self.cooling_rate
        else:
            self.T = self.T - (self.T0*100  / (self.iterations+1))


    def check_solution(self, new_configuration:ChargeCollection):
        '''
        Description
        -----------
        Checks the new solution (overall energy under new configuration). 
        
        If the new solution is better, accept it.
        If the new solution is worse, accept it with a probability depending on the temperature.

        Parameters
        ----------
        new_configuration : `ChargeCollection`
            The new charge configuration.
        '''
        new_energy = new_configuration.get_total_energy()
        old_energy = self.energy
        delta_energy = new_energy - old_energy

        # If new configuration is better, accept it
        if  delta_energy < 0:
            self.charges = new_configuration
            self.energy = new_energy
        # Else, if new configuration is worse, accept it given probability    
        elif random.random() < math.exp(-delta_energy / self.T):
            self.charges = new_configuration
            self.energy = new_energy


    def save_results(self, title):
        '''
        Saves energy and temperature data to csv.
        '''
        data = np.vstack((self.energy_arr, self.T_arr)).T
        np.savetxt(f"./data/{title}.csv", data, delimiter=',', header="energy, temperature")
        print(f"\nOutput saved to {title}")


    def run(self, iterations, neighbor='iterative', verbose=False, animate=False, save=False):
        '''
        Description
        -----------
        Runs the simulation.

        Parameters
        ----------
        iterations : `int`
            The number of iterations to run the simulation for.
        neighbor : `str`
            The neighbor selection method to use. Options are 'random', 'iterative', and 'by_force'.
        verbose : `bool`
            Whether to print the current iteration and energy.
        animate : `bool`
            Whether to animate the simulation.
        save : `bool`
            Whether to save the results to a csv file.
        '''
        self.iterations = iterations
        self.energy_arr = np.zeros(iterations)
        self.T_arr = np.zeros(iterations)

        if animate:
            plt.ion()
            plt.figure()

        for iteration in range(iterations):
            if verbose:
                print(f'Iteration {iteration + 1}/{iterations}, current energy: {self.energy}, T: {self.T}                ', end='\r')
            
            # Attempt N steps, one per charge TODO: other ways of picking charges?            
            charge_indices = self.rank_charges(neighbor=neighbor)
            for i in charge_indices:
                new_configuration = copy.deepcopy(self.charges)
                self.move_charge(new_configuration, charge_index=i)
                self.check_solution(new_configuration)  # accept or reject new configuration

            # Update the temperature for every 10th iteration
            if (iteration % self.chain_length == 0):
                self.update_temperature()

            if save:
                self.energy_arr[iteration] = self.energy
                self.T_arr[iteration] = self.T
            
            if animate:
                plt.clf()
                self.charges.plot_charges()
                plt.pause(0.005)
        
        plt.ioff() if animate else None 

        if save:
            self.save_results(title=f"{iteration + 1}_N_{self.charges.N}_iters_max_step_{self.max_stepsize}_T0_{self.T0}_cooling_{self.cooling_rate}")

        return self.charges
