import copy
import random 
import math
from charge_collection import ChargeCollection
import matplotlib.pyplot as plt 
import numpy as np

class HillClimber:

    def __init__(self, charges):
        if not charges.is_solution():
            raise Exception("HillClimber requires a valid configuration")
        
        self.charges = copy.deepcopy(charges)
        self.energy = charges.get_total_energy()
        self.max_stepsize = 0.7
        self.energy_arr = None 
        self.T_arr = None

    def mutate_configuration(self, new_configuration, charge_index):
        """
        Change the position of a given charge.
        """
        x_step = random.uniform(-self.max_stepsize, self.max_stepsize) 
        y_step = random.uniform(-self.max_stepsize, self.max_stepsize)
        new_configuration.change_charge_position(charge_index, x_step, y_step)

    def check_solution(self, new_configuration, iteration):
        """
        Check and accept better solution than current one.
        """
        new_energy = new_configuration.get_total_energy()
        old_energy = self.energy 

        if new_energy <= old_energy:
            self.charges = new_configuration
            self.energy = new_energy 

    def run(self, iterations, verbose=False, animate=False, save=False):
        """
        Runs the hillclimber algorithm for a specific amount of iterations.
        """
        self.iterations = iterations
        self.energy_arr = np.zeros(iterations * self.charges.N)
        self.T_arr = np.zeros(iterations * self.charges.N)

        try:
            self.T 
        except AttributeError:
            self.T = None

        if animate:
            plt.ion()
            plt.figure()
    
        for iteration in range(iterations):
            # Nice trick to only print if variable is set to True
            print(f'Iteration {iteration}/{iterations}, current energy: {self.energy}, T: {self.T}                ', end='\r') if verbose else None
            
            # attempt N mutations per iteration (one per charge)
            for charge_index in range(self.charges.N):

                # Create a copy of the configuration to simulate the change
                new_configuration = copy.deepcopy(self.charges)

                self.mutate_configuration(new_configuration, charge_index)

                # Accept it if new configuration is better
                self.check_solution(new_configuration, iteration)    

                # Save energy state
                self.energy_arr[iteration * self.charges.N + charge_index] = self.energy if save else None
                self.T_arr[iteration * self.charges.N + charge_index] = self.T if self.T and save else None

            if animate:
                plt.clf()
                self.charges.plot_charges()
                plt.pause(0.005)
        
        plt.ioff() if animate else None 

        # save to csv
        if save:
            T0 = self.T0 if self.T else "no_temp"
            self.save_results(title=f"{iteration + 1}_iters_max_step_{self.max_stepsize}_T0_{T0}")

        # return best solution found
        return self.charges     

    def save_results(self, title):
        """
        Saves energy and temperature data to csv.
        """
        data = np.vstack((self.energy_arr, self.T_arr)).T
        np.savetxt(f"./data/{title}.csv", data, delimiter=',', header="energy, temperature")
        print(f"\nOutput saved to {title}")


class SimulatedAnnealing(HillClimber):
    """
    The SimulatedAnnealing class that changes position of random charge.
    Each improvement or equivalent solution is kept for the next iteration.
    Sometimes accepts solutions that are \'worse\', depending on the current temperature.

    Most of the functions are similar to those of the HillClimber class, which is why
    we use that as a parent class.
    """
    def __init__(self, charges, temperature=750):
        # Use the init of the Hillclimber class
        super().__init__(charges)

        # Starting temperature and current temperature
        self.T0 = temperature
        self.T = temperature

    def update_temperature(self):
        """
        This function implements a exponential cooling scheme.
        Same one used in the article on canvas.
        """
        self.T = self.T * 0.95 #- (self.T0 / self.iterations / 1100)

    def check_solution(self, new_configuration, iteration):
        """
        Checks and accepts better solutions than the current solution.
        Sometimes accepts solutions that are worse, depending on the current
        temperature.
        """
        new_energy = new_configuration.get_total_energy()
        old_energy = self.energy

        # Calculate the probability of accepting this new graph
        delta = new_energy - old_energy

        # Pull a random number between 0 and 1 and see if we accept the configuration!
        if  delta < 0 or random.random() < math.exp(- delta / self.T):
            self.charges = new_configuration
            self.energy = new_energy

        # Update the temperature 
        if iteration % 100 == 0:
            self.update_temperature()
        if self.T < 0.01:
            self.max_stepsize = 0.1
            # self.max_stepsize *= 0.9


if __name__=="__main__":
    N = 11
    # N = 10
    random_charges = ChargeCollection(N)
    random_charges.plot_charges()

    hillclimber = HillClimber(random_charges)
    hillclimber.run(iterations=10, verbose=True, animate=True, save=True)

    hillclimber.charges.plot_charges()

    simulated_annealing = SimulatedAnnealing(random_charges)
    simulated_annealing.run(iterations=20000, verbose=True, animate=True, save=True)
    simulated_annealing.charges.plot_charges()