'''
This is the main file for the project.

This file will be used to orchestrate the modules. I.e. call the other
files and run the project.
'''

# Imports
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from charge_collection import ChargeCollection
from HillClimber import SimulatedAnnealing
from HillClimber import HillClimber


def run_simulated_annealing(plot=True, save=False):
    '''
    Runs one simulated annealing run.
    '''
    N = 11      # = number of charges
    c = 0.9     # = cooling rate (taken from paper)

    # Create charge collection:
    random_charges = ChargeCollection(N-1)  # N-1 because we add a charge at (0,0) during iterations (TODO: change this)
    simulated_annealing = SimulatedAnnealing(random_charges, max_stepsize=0.5, cooling_rate=c)
    simulated_annealing.run(iterations=5000,verbose=True, animate=False, save=save)

    simulated_annealing.charges.plot_charges(random_charges)
    plt.axis('off')
    # plt.savefig('N{} I{} c{}.png'.format(N,5000,c))  # TODO: change this
    plt.show()


def main():
    '''Main function'''
    run_simulated_annealing()


if __name__ == '__main__':
    main()
