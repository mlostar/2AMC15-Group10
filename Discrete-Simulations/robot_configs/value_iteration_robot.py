import random
import copy
import numpy as np
#DEFAULT_TILE_VALUE = 0

#tile_location_value = {}
#experiences = []



def robot_epoch(robot):

    possible_moves = list(robot.dirs.values())
    # Init empty value grid
    value_init = np.zeros_like(robot.grid.cells)
    # Calculate the values via value iteration
    value_grid = value_iteration(robot,value_init)
    # Find action values
    # One step lookahead to calculate the new values(value_grid) for all the possible moves.
    action_values = [value_grid[add_coords(robot.pos , move)] + robot.grid.cells[add_coords(robot.pos,move)] for move in possible_moves]
    # Find the best move by finding the action with highest value for this state 
    move = possible_moves[action_values.index(max(action_values))]
    # Find out how we should orient ourselves:
    new_orient = get_orientation_by_move(robot, move=move)

    move_robot(robot, new_orient)

    #############################################


    # if 1.0 in list(possible_tiles.values()) or 2.0 in list(possible_tiles.values()):
    #     # If we can reach a goal tile this move:
    #     if 2.0 in list(possible_tiles.values()):
    #         move = get_move_by_label(possible_tiles, label=2)
    #     # If we can reach a dirty tile this move:
    #     elif 1.0 in list(possible_tiles.values()):
    #         # Find the move that makes us reach the dirty tile:
    #         move = get_move_by_label(possible_tiles, label=1)
    #     else:
    #         assert False
    # # If we cannot reach a dirty tile:
    # else:
    #     # If we can no longer move:
    #     while not robot.move():
    #         # Check if we died to avoid endless looping:
    #         if not robot.alive:
    #             break
    #         # Decide randomly how often we want to rotate:
    #         times = random.randrange(1, 4)
    #         # Decide randomly in which direction we rotate:
    #         if random.randrange(0, 2) == 0:
    #             # print(f'Rotating right, {times} times.')
    #             for k in range(times):
    #                 robot.rotate('r')
    #         else:
    #             # print(f'Rotating left, {times} times.')
    #             for k in range(times):
    #                 robot.rotate('l')
    # # print('Historic coordinates:', [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])])


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


def value_iteration(robot,vals,theta = 0.01,gamma=0.3):

    #Using theta in the actual check breaks the code for me for some reason? Thats why its not used.
    iter = 0
    while True:
        grid_cells = copy.deepcopy(robot.grid.cells)
        grid_cells[robot.pos] = 0
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
                action_values = {'n':grid_cells[x,y-1]+gamma*vals[x,y-1],
                                 's':grid_cells[x,y+1]+gamma*vals[x,y+1],
                                 'e':grid_cells[x+1,y]+gamma*vals[x+1,y],
                                 'w':grid_cells[x-1,y]+gamma*vals[x-1,y]}
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
        if max_diff < 0.01 and iter > 0:
            break
        iter+=1
    # Return converged value grid
    return vals