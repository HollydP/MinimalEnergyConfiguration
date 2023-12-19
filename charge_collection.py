import numpy as np
import math 
import matplotlib.pyplot as plt 
import sys 

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
    
    
    def get_total_energy(self):
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
    

    def plot_charges(self,random_charges=[]):
        """
        Plots current charge configuration.
        """
        x_coords = [coord[0] for coord in self.charges]
        y_coords = [coord[1] for coord in self.charges]

        theta = np.linspace(0, 2*np.pi, 100)
        x, y = np.cos(theta), np.sin(theta)
        plt.plot(x, y, label='boundary')
        plt.plot(x_coords, y_coords, "ko")
        plt.show()
    

    def change_charge_position(self, charge, x_step, y_step):
        """
        Changes the position of a charge in the configuration.
        """
        new_x = self.charges[charge][0] + x_step
        new_y = self.charges[charge][1] + y_step 

        # place on border if step pushes charge out of bounds
        # TODO: this doesn't seem correct
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

if __name__=="__main__":
    N = 3
    configuration = ChargeCollection(N)
    print(configuration.get_total_energy())
    print(configuration.charges)
    configuration.plot_charges()
