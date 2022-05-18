import copy
import numpy as np
import time
from collections import deque
import random
from typing import List, Tuple, Dict, Union
epsilon = 0.1
gamma = 0.90

CLEAN_REWARD = 0
DIRTY_REWARD = 1
GOAL_REWARD = 3
DEATH_REWARD = -10
OBSTACLE_REWARD = -1
WALL_REWARD = -1

# State     Label
# obstacle  -2
# wall      -1
# clean     0
# dirty     1
# goal      2
# death     3


def robot_epoch(robot):
    # (move_orientation, (move_increment)) pairs
    move_pairs = list(robot.dirs.items()) #[('n', (0,0))]
    # grid is (19,14), n_cols = 19, n_rows = 14
    # grid.flatten(order='F')?
    # no of rows
    n_rows = robot.grid.n_rows
    # no of cols
    n_cols = robot.grid.n_cols
    # total number of tiles
    n_tiles = n_rows * n_cols
    # Initialization of e-soft policy, Q and returns
    Q = np.zeros((n_tiles, 4)).tolist()
    N = np.zeros((n_tiles, 4)).tolist()
    D = np.zeros((n_tiles, 4)).tolist()

    pi = mk_greedy_policy(Q)                        # Target policy
    mu = ((np.ones((n_tiles, 4)) * epsilon) / 4)    # Behavior policy
    current_pos = robot.pos
    rewards = [CLEAN_REWARD,
               DIRTY_REWARD,
               GOAL_REWARD,
               DEATH_REWARD,
               OBSTACLE_REWARD,
               WALL_REWARD]
    grid = robot.grid.cells.flatten(order='F')
    grid[coord2ind(robot.pos)] = 0 # label for clean tile
    grid_rewards = grid_to_rewards(grid, rewards)

    # Repeat till target policy and Q values converge
    iter = 0
    while True:
        # Generate episode with behavior policy mu
        ep_history, ep_returns = generate_episode(mu, robot)

        # Check latest time/index action_mu != action_pi which is tau
        tau = len(ep_history)
        for i in np.arange(len(ep_history)-1, -1, -1):
            ind, action = ep_history[i]
            if action != np.argmax(pi[ind]):
                tau = i
                continue
        # For each pair s, a after index tau and tau itself
        for i in np.arange(tau, len(ep_history)):
            s, a = ep_history[i]
            # Gt
            Gt = ep_returns[i]
            # Calculate W
            W = 1
            for j in np.arange(i+1, len(ep_history)-1):
                s_k, a_k = ep_history[j]
                W = W * mu[s_k][a_k]
            N[s][a] = N[s][a] + W*Gt
            D[s][a] = D[s][a] + W
            Q[s][a] = N[s][a] + D[s][a]
            # For each state update policy
        prev_pi = pi
        pi = mk_greedy_policy(Q)
        iter += 1
        # Add iter >=5 check to at least do 5 walks
        if (np.array_equal(prev_pi, pi) and iter >= 5) or iter == 200:
            print("Iters until convergence: ", iter)
            break

    # Find next move based on converged policy
    current_pos = robot.pos
    next_move_id = np.argmax(pi[coord2ind(current_pos, n_rows)])
    next_orientation = move_pairs[next_move_id][0]
    # Move
    move_robot(robot, next_orientation)


def generate_episode(mu, robot, current_pos, move_pairs, grid_rewards, MAX_EP_LENGTH_R=400, MAX_EP_LENGTH=1000):
    # (move_orientation, (move_increment)) pairs
    # (position,action_orientation) pairs in order
    history = deque()
    # Stores returns of (position,action_orientation) pairs
    episode_returns = deque()
    is_episode_terminated = False
    rewards = copy.deepcopy(grid_rewards)
    # Run episode
    while not is_episode_terminated:
        # Find best move from policy
        chosen_move_idx = get_move_from_policy(mu, current_pos)
        move_orientation, move_increment = move_pairs[chosen_move_idx]
        next_pos = coordinate_finder(current_pos, move_increment)
        reward = rewards[next_pos]
        original_reward = grid_rewards[next_pos]

        # Ignore moves into walls/obstacles
        if original_reward != -1:
            # Add the move to history of moves
            history.append((current_pos, chosen_move_idx, reward))
            # Add reward of the move to returns of all previous moves and append
            episode_returns.append(reward)
            # Set current position to the next position if move was not into a wall/obstacle
            current_pos = next_pos
            # Set reward to zero since it's now a clean tile
            rewards[current_pos] = 0
        # Terminate the episode once the robot makes certain no of moves
        # or if the robot steps on a death cell
        ep_length = len(episode_returns)
        if (ep_length >= MAX_EP_LENGTH_R and sum(episode_returns) > 0) \
            or ep_length == MAX_EP_LENGTH \
            or grid_rewards[current_pos] == DEATH_REWARD:
            print("Simulated moves: ",len(episode_returns))
            # ep_length long array of gamma values
            gammas = np.full((ep_length,), gamma)
            powers = np.arange(ep_length)
            # Calculate the decaying gamma values
            gammas = np.power(gammas, powers)
            ep_rewards = np.array(episode_returns)
            # Calculate returns by doing a dot product between the array of rewards after this move
            # with decaying gamma values
            episode_returns = [np.dot(gammas[:ep_length-i], ep_rewards[i:]) for i in range(ep_length)]
            is_episode_terminated = True
    return history, episode_returns


def get_move_from_policy(policy, current_pos):
    indices = [0, 1, 2, 3]  # move indices
    # Choose next move based on policy probabilities
    move_idx = random.choices(indices, weights=policy[current_pos], k=1)[0]
    return move_idx


def grid_to_rewards(grid, rewards):
    """
    Returns a reward grid based on the input robot.
    """
    return np.array(list(map(lambda x: rewards[x], grid)))


def add_coords(c1, c2):
    """
    Return the two input coords added together.
    """
    return (c1[0] + c2[0], c1[1] + c2[1])


def coordinate_finder(pos, direction, as_ind=True):
    """
    Given an initial position and move direction calculate the next position
    calculate the next position based on the given current position and move direction
    """
    if len(pos) != 2:
        pos = ind2coord(pos)
    new_pos = add_coords(pos, direction)
    if as_ind:
        new_pos = coord2ind(new_pos)
    return new_pos


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


def ind2coord(ind: int, n_row: int) -> Tuple[int, int]:
    x = ind % n_row
    y = ind // n_row
    return int(x), int(y)


def coord2ind(coord: Tuple[int, int], n_row: int) -> int:
    return coord[0] + n_row*coord[1]


def mk_greedy_policy(Q):
    pi = np.zeros_like(Q)
    pi[:, np.argmax(Q, axis=1)] = 1
    return pi

