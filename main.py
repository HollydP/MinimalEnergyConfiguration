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

# from charge_collection import ChargeCollection
# from HillClimber import SimulatedAnnealing
# from HillClimber import HillClimber

from classes import ChargeCollection, SimulatedAnnealing

def run_simulated_annealing(iterations=5000, plot=False, save=False, verbose=False):
    '''
    Runs one simulated annealing run.
    '''
    N = 12      # = number of charges
    c = 0.1     # = cooling rate (lower = faster cooling, but more iterations near zero)

    # Create charge collection:
    random_charges = ChargeCollection(N)  # N-1 because we add a charge at (0,0) during iterations (TODO: change this)
    simulated_annealing = SimulatedAnnealing(random_charges, max_stepsize=1.0, cooling_rate=c)
    simulated_annealing.run(iterations=iterations, verbose=verbose, animate=False, save=save)

    if plot:
        simulated_annealing.charges.plot_charges(random_charges)
    
    return simulated_annealing.energy


def main():
    '''Main function'''
    iterations = 10_000
    min_energy = run_simulated_annealing(iterations=iterations, plot=True, verbose=True)
    print('')
    print(f"E_min = {min_energy} ({iterations} iterations)")


if __name__ == '__main__':
    main()
