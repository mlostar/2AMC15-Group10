from copy import deepcopy
import numpy as np

DISCOUNT_FACTOR = 0.9
THETA = 0.01

CLEAN_REWARD = 0
DIRTY_REWARD = 1
GOAL_REWARD = 3
DEATH_REWARD = -10
OBSTACLE_REWARD = 0
WALL_REWARD = 0

# obstacle  -2
# wall      -1
# clean     0
# dirty     1
# goal      2
# death     3


def robot_epoch(robot):
    # Initialization

    possible_directions = list(robot.dirs.values())

    V = np.zeros_like(robot.grid.cells)  # initialize V
    policy = 1/4 * np.ones((robot.grid.n_cols, robot.grid.n_rows, 4)) # initialize equiprobably policy
    policy_evaluation(robot, V, policy)
    
def policy_evaluation(robot, V, policy):
    i = 0
    while True:
        i += 1
        delta = 0

        grid_cells = deepcopy(robot.grid.cells)
        grid_cells[robot.pos] = 0
        n_rows = robot.grid.n_rows
        n_cols = robot.grid.n_cols

        for x in range(1, n_cols-1):
            for y in range(1, n_rows-1):
                # Robot cant visit negative tiles
                if grid_cells[x, y] in [-1, -2, -3]:
                    continue

                v = 0
                v += policy[x,y,0] * (grid_cells[x, y-1]+DISCOUNT_FACTOR*V[x, y-1]) # north
                v += policy[x,y,1] * (grid_cells[x, y+1]+DISCOUNT_FACTOR*V[x, y+1]) # south
                v += policy[x,y,2] * (grid_cells[x+1, y]+DISCOUNT_FACTOR*V[x+1, y]) # east
                v += policy[x,y,3] * (grid_cells[x-1, y]+DISCOUNT_FACTOR*V[x-1, y]) # west

                delta = max(delta, np.abs(V[x][y] - v))

                V[x][y] = v

        if delta < THETA:
            break
    print(f"Policy evaluation convergence: {i} iterations")