from copy import deepcopy
import math
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
    possible_moves = list(robot.dirs.values())

    V = np.zeros_like(robot.grid.cells)  # initialize V
    policy = 1/4 * np.ones((robot.grid.n_cols, robot.grid.n_rows, 4)) # initialize equiprobably policy

    i = 0
    policy_stable = False
    while not policy_stable:
        i += 1
        # Policy evaluation
        policy_evaluation(robot, V, policy)

        #Policy improvement
        policy_stable = policy_improvment(robot, V, policy)

    # print(f"Policy iteration convergence: {i} iterations")

    action_values = [V[add_coords(robot.pos , move)] + robot.grid.cells[add_coords(robot.pos,move)] for move in possible_moves]
    # Find the best move by finding the action with highest value for this state
    move = possible_moves[action_values.index(max(action_values))]
    # Find out how we should orient ourselves:
    new_orient = get_orientation_by_move(robot, move=move)

    move_robot(robot, new_orient)

        
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
    # print(f"Policy evaluation convergence: {i} iterations")

def policy_improvment(robot, V, policy):
    policy_stable = True # assume stable at first

    grid_cells = deepcopy(robot.grid.cells)
    grid_cells[robot.pos] = 0
    n_rows = robot.grid.n_rows
    n_cols = robot.grid.n_cols


    for x in range(1, n_cols-1):
        for y in range(1, n_rows-1):
            old_policy = policy[x,y,:].copy()
            best_actions = []
            max_v = -math.inf
            
            # Check best actions
            action_values = [grid_cells[x,y-1]+DISCOUNT_FACTOR*V[x,y-1],
                            grid_cells[x,y+1]+DISCOUNT_FACTOR*V[x,y+1],
                            grid_cells[x+1,y]+DISCOUNT_FACTOR*V[x+1,y],
                            grid_cells[x-1,y]+DISCOUNT_FACTOR*V[x-1,y]]

            for action, value in enumerate(action_values):
                if value > max_v:
                    max_v = value
                    best_actions = [action]
                elif value == max_v: # handle possible same action value
                    best_actions.append(action)

            # Create new policy
            prob = 1/len(best_actions)
            for action in range(len(action_values)):
                if action in best_actions:
                    policy[x,y,action] = prob
                else:
                    policy[x,y,action] = 0

            # Check if policy changed
            if not (old_policy == policy[x,y,:]).all():
                policy_stable = False

    return policy_stable

def move_robot(robot, direction):
    # Orient ourselves towards the dirty tile:
    while direction != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    # Move:
    robot.move()

# Returns one of the directions (e.g. 'n') given the move
def get_orientation_by_move(robot, move):
    return list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]

def add_coords(c1, c2):
    return (c1[0]+c2[0], c1[1]+c2[1])