import numpy as np
import math 
import matplotlib.pyplot as plt 
import sys 

class ChargeCollection:

    def __init__(self, N) -> None:
        self.N = N
        self.charges = self.initialize_charge_arrangement()

    def initialize_charge_arrangement(self):
        """"
        Set up starting configuration of charges.
        """
        r = np.sqrt(np.random.uniform(0, 1, self.N))
        theta = np.random.uniform(0, 2 * math.pi, self.N)

        x = r * np.cos(theta) 
        y = r * np.sin(theta)
        
        return [[x[i], y[i]] for i in range(self.N)]
    
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
    
    def plot_charges(self,random_charges=[]):
        """
        Plots current charge configuration.
        """
        x_coords = [coord[0] for coord in self.charges]
        y_coords = [coord[1] for coord in self.charges]

        theta = np.linspace(0, 2*np.pi, 100)
        x, y = np.cos(theta), np.sin(theta)
        plt.plot(x, y, label='boundary')
        plt.Circle((0,0), 1, edgecolor='b')
        plt.plot(x_coords, y_coords, "o")
        if len(random_charges) > 0 :
            x_initial = [coord[0] for coord in random_charges]
            y_initial = [coord[1] for coord in random_charges]
            plt.plot(x_initial, y_initial, "o",c = 'grey')
        plt.show()
    
    def change_charge_position(self, charge, x_step, y_step):
        """"
        Changes the position of a charge in the configuration.
        """
        self.charges[charge][0] += x_step
        self.charges[charge][1] += y_step 

    def is_solution(self):
        """"
        Checks if the charge configuration is valid.
        """
        for coord in self.charges:
            if math.dist([0, 0], coord) > 1:
                return False
        return True

if __name__=="__main__":
    N = 3
    configuration = ChargeCollection(N)
    print(configuration.get_total_energy())
    print(configuration.charges)
    configuration.plot_charges()
