import copy
import random

import numpy as np


random.seed(0)
np.random.seed(0)


def robot_epoch(robot, gamma=0.1, epsilon=1.0, alpha=0.99, n_epochs=500, max_episode_length=50):
    """
    Run an epoch of the value iteration robot.
    """

    # Init the policy
    pi_init = np.zeros(shape=(*np.array(robot.grid.cells).shape, 4))
    # Calculate the values via value iteration
    optimal_qs = estimate_q(robot,
                            gamma=gamma,
                            epsilon=epsilon,
                            alpha=alpha,
                            n_epochs=n_epochs,
                            max_episode_length=max_episode_length)
    # Find optimal move
    move = get_optimal_move(robot, optimal_qs)
    # Find out how we should orient ourselves:
    new_orient = get_orientation_by_move(robot, move=move)

    move_robot(robot, new_orient)


def get_optimal_move(robot, pi):
    """
    Return the optimal move.
    """
    possible_moves = list(robot.dirs.items())
    return possible_moves[np.argmax(pi[robot.pos])][1]


def get_orientation_by_move(robot, move):
    """
    Return the orientation with the given move.
    """
    return list(robot.dirs.keys())[list(robot.dirs.values()).index(move)]


def move_to_index(robot, move):
    """
    Returns the index of the move.
    """
    return list(robot.dirs.values()).index(move)


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


def get_epsilon_greedy_move(robot, epsilon, q):
    """
    Returns the epsilon greedy move.
    """
    if random.random() > epsilon:
        return get_optimal_move(robot, q)
    else:
        return random.choice(list(robot.dirs.values()))


def estimate_q(robot,
               gamma=0.3,
               epsilon=0.1,
               epsilon_min=0.01,
               epsilon_decay=0.99,
               alpha=0.9,
               n_epochs=50,
               max_episode_length=100):
    """
    Estimate the optimal q values.
    """

    q = np.zeros(shape=(*np.array(robot.grid.cells).shape, 4))
    current_epsilon = epsilon
    rewards_per_cell = grid_to_rewards(robot)

    for i in range(n_epochs):
        current_epsilon = max(current_epsilon*epsilon_decay, epsilon_min)
        robot_copy = copy.deepcopy(robot)

        for j in range(max_episode_length):
            # Make a move and record the details
            move = get_epsilon_greedy_move(robot_copy, current_epsilon, q)
            old_position = robot_copy.pos
            new_orient = get_orientation_by_move(robot_copy, move=move)
            move_robot(robot_copy, new_orient)
            new_position = robot_copy.pos
            reward = rewards_per_cell[add_coords(old_position, move)]

            # Look ahead for the next move and its q value
            move_index = move_to_index(robot_copy, move)
            next_move = get_epsilon_greedy_move(robot_copy, current_epsilon, q)
            next_move_q = q[new_position][move_to_index(robot_copy, next_move)]

            # TD equation
            q[old_position][move_index] += alpha * \
                (reward+gamma*next_move_q-q[old_position][move_index])

            # Early stop if the robot has died or cleaned all the cells.
            if not robot_copy.alive or (robot_copy.grid.cells >= 1).sum() == 0:
                break

    return q
