'''
This is the main file for the project.

This file will be used to orchestrate the modules. I.e. call the other
files and run the project.
'''

# Imports
from classes import ChargeCollection, SimulatedAnnealing

def main():
    '''
    Runs one simulated annealing run.
    '''
    iterations = 10_000

    N = 12      # = number of charges
    c = 0.9     # = cooling rate (lower = faster cooling, but more iterations near zero)

    simulated_annealing = SimulatedAnnealing(
        charges=ChargeCollection(N), 
        max_stepsize=0.5, 
        chain_length = 10,
        cooling_rate=c,
        init_temperature=100
    )
    
    simulated_annealing.run(
        iterations=iterations, 
        verbose=True, 
        animate=False, 
        save=False
    )

    simulated_annealing.charges.plot_charges()

    min_energy = simulated_annealing.energy
    
    print('')
    print(f"E_min = {min_energy} ({iterations} iterations)")


if __name__ == '__main__':
    main()
