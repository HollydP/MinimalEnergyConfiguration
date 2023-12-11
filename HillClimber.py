import copy
import random 
import math
from charge_collection import ChargeCollection

class HillClimber:

    def __init__(self, charges):
        if not charges.is_solution():
            raise Exception("HillClimber requires a valid configuration")
        
        self.charges = copy.deepcopy(charges)
        self.energy = charges.get_total_energy()
        self.max_stepsize = 0.7

    def mutate_configuration(self, new_configuration):
        """
        Change the position of a random charge.
        """
        random_index = random.randint(0, self.charges.N - 1)
        x_step = random.random() * 2 * self.max_stepsize - self.max_stepsize
        y_step = random.random() * 2 * self.max_stepsize - self.max_stepsize
        new_configuration.change_charge_position(random_index, x_step, y_step)

    def check_solution(self, new_configuration, iteration):
        """
        Check and accept better solution than current one.
        """
        new_energy = new_configuration.get_total_energy()
        old_energy = self.energy 

        if new_energy <= old_energy:
            self.charges = new_configuration
            self.energy = new_energy 

    def run(self, iterations, verbose=False):
        """
        Runs the hillclimber algorithm for a specific amount of iterations.
        """
        self.iterations = iterations

        for iteration in range(iterations):
            # Nice trick to only print if variable is set to True
            print(f'Iteration {iteration}/{iterations}, current energy: {self.energy}                ', end='\r') if verbose else None

            # Create a copy of the configuration to simulate the change
            new_configuration = copy.deepcopy(self.charges)

            self.mutate_configuration(new_configuration)

            # Accept it if new configuration is better
            self.check_solution(new_configuration, iteration)    

class SimulatedAnnealing(HillClimber):
    """
    The SimulatedAnnealing class that changes position of random charge.
    Each improvement or equivalent solution is kept for the next iteration.
    Sometimes accepts solutions that are \'worse\', depending on the current temperature.

    Most of the functions are similar to those of the HillClimber class, which is why
    we use that as a parent class.
    """
    def __init__(self, charges, temperature=5000):
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
        self.T = self.T * 0.9 #- (self.T0 / self.iterations / 1100)

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
        if iteration % 1100 == 0:
            self.update_temperature()
            self.max_stepsize *= 0.99


if __name__=="__main__":
    N = 11
    random_charges = ChargeCollection(N)
    random_charges.plot_charges()

    # hillclimber = HillClimber(random_charges)
    # hillclimber.run(iterations=100000, verbose=True)

    # hillclimber.charges.plot_charges()

    simulated_annealing = SimulatedAnnealing(random_charges)
    simulated_annealing.run(iterations=150000, verbose=True)
    simulated_annealing.charges.plot_charges()