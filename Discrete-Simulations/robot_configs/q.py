import copy
import random

import numpy as np


random.seed(0)
np.random.seed(0)


def robot_epoch(robot, gamma=0.9, epsilon=0.5, alpha=0.3, n_epochs=100, max_episode_length=200):
    """
    Run an epoch of the value iteration robot.
    """

    # Init empty q values cube
    q_init = np.zeros(shape=(*np.array(robot.grid.cells).shape, 4))
    # Calculate the values via value iteration
    optimal_qs = estimate_qs(robot,
                             q_init,
                             gamma=gamma,
                             epsilon=epsilon,
                             alpha=alpha,
                             n_epochs=n_epochs,
                             max_episode_length=max_episode_length)
    # Find optimal move
    # TODO: filter out illegal moves
    move = get_optimal_move(robot, optimal_qs)
    # Find out how we should orient ourselves:
    new_orient = get_orientation_by_move(robot, move=move)

    move_robot(robot, new_orient)


def get_optimal_move(robot, optimal_qs):
    possible_moves = list(robot.dirs.items())
    return possible_moves[np.argmax(optimal_qs[robot.pos])][1]


def get_orientation_by_move(robot, move):
    """
    Return the orientation with the given move.
    """
    return list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]


def move_robot(robot, direction):
    """
    Rotates the robot into the given direction and calls the move function.
    """
    # Orient ourselves towards the dirty tile:
    while direction != robot.orientation:
        # If we don't have the wanted orientation, rotate clockwise until we do:
        # print('Rotating right once.')
        robot.rotate('r')
    # Move:
    robot.move()


def add_coords(c1, c2):
    """
    Return the two input coords added together.
    """
    return (c1[0]+c2[0], c1[1]+c2[1])


def grid_to_rewards(robot):
    """
    Returns a reward grid based on the input robot.
    """
    temp_grid = copy.deepcopy(robot.grid.cells)
    temp_grid[robot.pos] = 0
    values = {-2: -2,
              -1: -1,
              0: 0,
              1: 1,
              2: 2,
              3: -3}
    output = np.vectorize(values.__getitem__)(temp_grid)

    return output


def estimate_qs(robot,
                qs,
                theta=0.01,
                gamma=0.3,
                epsilon=0.1,
                alpha=0.9,
                n_epochs=50,
                max_episode_length=100):
    """
    Estimate the optimal q values.
    """

    q_n = np.transpose(qs[:, :, 0], axes=(1, 0))
    q_e = np.transpose(qs[:, :, 1], axes=(1, 0))
    q_s = np.transpose(qs[:, :, 2], axes=(1, 0))
    q_w = np.transpose(qs[:, :, 3], axes=(1, 0))

    for i in range(n_epochs):
        robot_copy = copy.deepcopy(robot)

        for j in range(max_episode_length):
            rewards_per_cell = grid_to_rewards(robot_copy)

            if random.random() > epsilon:
                move = get_optimal_move(robot_copy, qs)
            else:
                move = random.choice(list(robot_copy.dirs.values()))

            old_position = robot_copy.pos
            new_orient = get_orientation_by_move(robot_copy, move=move)
            move_robot(robot_copy, new_orient)

            new_position = robot_copy.pos
            reward = rewards_per_cell[add_coords(old_position, move)]

            move_index = list(robot_copy.dirs.values()).index(move)

            # TD equation
            qs[old_position][move_index] += alpha*(reward+gamma*(np.max(qs[new_position]))-qs[old_position][move_index])

            # Early stop if the robot has died or cleaned all the cells.
            if not robot_copy.alive or (robot_copy.grid.cells >= 1).sum() == 0:
                break

            # rewards_per_cell = grid_to_rewards(robot)
            transposed_reward = np.transpose(rewards_per_cell, axes=(1, 0))

    return qs

