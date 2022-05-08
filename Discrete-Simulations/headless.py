# Import our robot algorithm to use in this simulation:
from copy import copy

import numpy as np
import pathlib
from robot_configs.value_iteration_robot import robot_epoch
import pickle
from environment import Robot
import matplotlib.pyplot as plt
from itertools import product
import random
import multiprocessing as mp


ALGORITHM = "value-iteration-test"
random.seed(1)


def rerun(n=1, grid_file="house.grid", gamma=0.3, random_move_prob=0.0):
    # Cleaned tile percentage at which the room is considered 'clean':
    stopping_criteria = 100

    # Keep track of some statistics:
    efficiencies = []
    n_moves = []
    deaths = 0
    cleaned = []

    # Run n times:
    for i in range(n):
        # Open the grid file.
        # (You can create one yourself using the provided editor).
        with open(f'grid_configs/{grid_file}', 'rb') as f:
            grid = pickle.load(f)
        # Calculate the total visitable tiles:
        n_total_tiles = (grid.cells >= 0).sum()
        # Spawn the robot at (1,1) facing north with battery drainage enabled:
        robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2, p_move=random_move_prob)
        # Keep track of the number of robot decision epochs:
        n_epochs = 0

        while True:
            n_epochs += 1
            # Do a robot epoch (basically call the robot algorithm once):
            robot_epoch(robot, gamma=gamma)
            # Stop this simulation instance if robot died :( :
            if not robot.alive:
                deaths += 1
                break
            # Calculate some statistics:
            clean = (grid.cells == 0).sum()
            dirty = (grid.cells >= 1).sum()
            goal = (grid.cells == 2).sum()
            # Calculate the cleaned percentage:
            clean_percent = (clean / (dirty + clean)) * 100
            # See if the room can be considered clean, if so, stop the simulaiton instance:
            if clean_percent >= stopping_criteria and goal == 0:
                break
            # Calculate the effiency score:
            moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
            u_moves = set(moves)
            n_revisted_tiles = len(moves) - len(u_moves)
            efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)

        # Keep track of the last statistics for each simulation instance:
        efficiencies.append(float(efficiency))
        n_moves.append(len(robot.history[0]))
        cleaned.append(clean_percent)

    # Make some plots:
    plt.figure(figsize=(10, 10))
    plt.hist(cleaned)
    plt.title(f'Percentage of tiles cleaned -- grid: {grid_file} -- gamma: {gamma} -- random move: {random_move_prob}')
    plt.xlabel('% cleaned')
    plt.ylabel('count')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.hist(efficiencies)
    plt.title(f'Efficiency of robot -- grid: {grid_file} -- gamma: {gamma} -- random move: {random_move_prob}')
    plt.xlabel('Efficiency %')
    plt.ylabel('count')
    plt.show()

    return np.median(cleaned), np.median(efficiencies)


# Append a new result to a dictionary of results which is assumed to have keys for the metrics
def append_new_result(runs_output, original_combination):
    print(id(original_combination))
    median_cleaned, median_efficiency = runs_output
    print(f"For {original_combination}, mc: {median_cleaned} me: {median_efficiency}")

    # Insert the parameters
    for key, value in original_combination.items():
        results[key].append(value)

    # Insert the result
    results["median cleaned percentage"].append(median_cleaned)
    results["median efficiency percentage"].append(median_efficiency)


if __name__ == "__main__":
    results = {"median cleaned percentage": [],
               "median efficiency percentage": []}
    # Make sure that the key matches the name of the argument
    parameters = {"grid_file": [path.name for path in pathlib.Path("grid_configs").glob("*")][:],
                  "random_move_prob": [0.0, 0.5],
                  "gamma": [0.3, 0.5, 0.7]}

    # Make a cartesian product of the parameter values and place them in a list of dictionaries where the keys are the
    # parameter names and the values are taken from the respective parameter combination
    combinations = product(*parameters.values())
    combinations = [{param_name: param_value for param_name, param_value in zip(parameters.keys(), param_values)}
                    for param_values in combinations]

    # Prepare the entries for the parameters in the results
    for key in parameters.keys():
        results[key] = []

    # Run the model multiple times for each combination of the parameters in parallel
    pool = mp.Pool()

    for combination in combinations:
        # Need this awkward setup due to late binding in Python
        def fun(combi=combination):
            pool.apply_async(rerun, kwds=combination, callback=lambda out: append_new_result(out, combi))

        fun(combination)

    pool.close()
    pool.join()
