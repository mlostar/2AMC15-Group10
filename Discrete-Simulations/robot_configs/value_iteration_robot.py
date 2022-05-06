import random
import copy
import numpy as np


def robot_epoch(robot,gamma=0.3):

    possible_moves = list(robot.dirs.items())
    # Init empty value grid
    value_init = np.zeros_like(robot.grid.cells)
    # Reward grid
    grid_cells = grid_to_rewards(robot)
    # Calculate the values via value iteration
    value_grid = value_iteration(robot,value_init, gamma=gamma)
    # Find action values
    # One step lookahead to calculate the new values(value_grid) for all the possible moves.

    action_values = [calc_action_val(move_orientation,
                                     robot.pos,
                                     grid_cells,
                                     value_grid,
                                     gamma,
                                     robot.p_move)
                     for move_orientation,move_increment in possible_moves]
    # Find the best move by finding the action with highest value for this state
    move = possible_moves[action_values.index(max(action_values))][1]
    # Find out how we should orient ourselves:
    new_orient = get_orientation_by_move(robot, move=move)

    move_robot(robot, new_orient)


def get_move_by_label(tiles, label):
    return list(tiles.keys())[list(tiles.values()).index(label)]


# Returns one of the directions (e.g. 'n') given the move
def get_orientation_by_move(robot, move):
    return list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]


def move_robot(robot, direction):
    # Orient ourselves towards the dirty tile:
    while direction != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    # Move:
    robot.move()

def add_coords(c1, c2):
    return (c1[0]+c2[0], c1[1]+c2[1])


# Given an initial position and move direction calculate the next position
# calculate the next position based on the given current position and move direction 
def coordinate_finder(pos,direction):
    if direction == 'n':
        return (pos[0],pos[1]-1)
    elif direction == 's':
        return (pos[0],pos[1]+1)
    elif direction == 'e':
        return (pos[0]+1,pos[1])
    elif direction == 'w':
        return (pos[0]-1,pos[1])


# Convert robot grid to a value grid
def grid_to_rewards(robot):
    temp_grid = copy.deepcopy(robot.grid.cells)
    temp_grid[robot.pos] = 0
    values = {-2:-2,
              -1:-1,
              0:0,
              1:1,
              2:2,
              3:-3}
    return np.vectorize(values.__getitem__)(temp_grid)


def calc_action_val(action_orientation,pos,grid_cells,vals,gamma,random_move_p):

    x = pos[0]
    y = pos[1]
    # Calculate action value based on random move probability
    # Primary move probability = 1-p_move + p_move/4
    # Secondary move probability = p_move/4
    if action_orientation == 'n':
        r_move_n = ((1-random_move_p)+(random_move_p/4))*(grid_cells[x, y - 1] + gamma * vals[x, y - 1])
        r_move_s = (random_move_p/4)*(grid_cells[x,y+1]+gamma*vals[x,y+1])
        r_move_e = (random_move_p/4)*(grid_cells[x+1,y]+gamma*vals[x+1,y])
        r_move_w = (random_move_p/4)*(grid_cells[x-1,y]+gamma*vals[x-1,y])
    elif action_orientation == 's':
        r_move_n = (random_move_p / 4) * (grid_cells[x, y - 1] + gamma * vals[x, y - 1])
        r_move_s = ((1 - random_move_p) + (random_move_p / 4)) * (grid_cells[x, y + 1] + gamma * vals[x, y + 1])
        r_move_e = (random_move_p / 4) * (grid_cells[x + 1, y] + gamma * vals[x + 1, y])
        r_move_w = (random_move_p / 4) * (grid_cells[x - 1, y] + gamma * vals[x - 1, y])
    elif action_orientation == 'e':
        r_move_n = (random_move_p / 4) * (grid_cells[x, y - 1] + gamma * vals[x, y - 1])
        r_move_s =  (random_move_p / 4) * (grid_cells[x, y + 1] + gamma * vals[x, y + 1])
        r_move_e = ((1 - random_move_p) + (random_move_p / 4)) * (grid_cells[x + 1, y] + gamma * vals[x + 1, y])
        r_move_w = (random_move_p / 4) * (grid_cells[x - 1, y] + gamma * vals[x - 1, y])
    elif action_orientation == 'w':
        r_move_n = (random_move_p / 4) * (grid_cells[x, y - 1] + gamma * vals[x, y - 1])
        r_move_s = (random_move_p / 4) * (grid_cells[x, y + 1] + gamma * vals[x, y + 1])
        r_move_e = (random_move_p / 4) * (grid_cells[x + 1, y] + gamma * vals[x + 1, y])
        r_move_w = ((1 - random_move_p) + (random_move_p / 4)) * (grid_cells[x - 1, y] + gamma * vals[x - 1, y])

    return r_move_n + r_move_s + r_move_e + r_move_w


def value_iteration(robot,vals,theta = 0.01,gamma=0.3):

    #Using theta in the actual check breaks the code for me for some reason? Thats why its not used.

    while True:
        grid_cells = grid_to_rewards(robot)

        n_rows = robot.grid.n_rows
        n_cols = robot.grid.n_cols
        max_diff = 0
        for x in range(1,n_cols-1):
            for y in range(1,n_rows-1):
                # Robot cant visit negative tiles
                if grid_cells[x,y] in [-1,-2,-3]:
                    continue
                # Calculate values (action return + next state value)
                # a dictionary with four directions as keys and their corresponding action returns $+$ next state values as values
                # TODO: Probability calculations
                action_values = {'n':calc_action_val('n',(x,y),grid_cells,vals,gamma,robot.p_move),
                                 's':calc_action_val('s',(x,y),grid_cells,vals,gamma,robot.p_move),
                                 'e':calc_action_val('e',(x,y),grid_cells,vals,gamma,robot.p_move),
                                 'w':calc_action_val('w',(x,y),grid_cells,vals,gamma,robot.p_move)}
                # Get the orientation of the best action
                # The key with the maximum value among all keys in action values
                best_action = max(action_values.keys(), key=(lambda key: action_values[key]))
                # Find next position after action
                next_pos = coordinate_finder((x,y),best_action)
                # Val += action_return + val_of_next_pos
                difference = grid_cells[next_pos] + gamma*vals[next_pos] - vals[x,y]
                vals[x,y] = grid_cells[next_pos] + gamma*vals[next_pos]
                if difference > max_diff:
                    max_diff = difference
        if max_diff < theta:
            break
    # Return converged value grid
    return vals