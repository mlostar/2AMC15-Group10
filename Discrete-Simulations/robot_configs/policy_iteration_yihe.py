import math
import numpy as np
import copy
from typing import Tuple, Dict, List, Union

DISCOUNT_FACTOR = 0.9
THETA = 10
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
    # Initialization
    n_rows = robot.grid.n_rows
    n_cols = robot.grid.n_cols
    possible_moves = list(robot.dirs.values())
    robot.grid.cells[robot.pos] = 0
    rewards = [CLEAN_REWARD,
               DIRTY_REWARD,
               GOAL_REWARD,
               DEATH_REWARD,
               OBSTACLE_REWARD,
               WALL_REWARD] # TODO: put this as global var
    V = np.zeros(n_rows*n_cols) # initialize V
    policy = 1 / 4 * np.ones((robot.grid.n_cols*robot.grid.n_rows, 4)) # initialize equal probability policy

    policy_stable = False
    while not policy_stable:
        # Policy evaluation
        policy_evaluation(robot, V, policy, rewards, possible_moves)

        # Policy improvement
        policy_stable = policy_improvement(robot, V, policy, rewards, possible_moves)

    # Find the best move by finding the action with highest value for this state
    move = possible_moves[np.argmax(policy[np.coord2ind(robot.pos, n_rows)])]
    # Find out how we should orient ourselves:
    new_orient = get_orientation_by_move(robot, move=move)

    move_robot(robot, new_orient)


def ind2coord(ind: int, n_row: int) -> Tuple[int, int]:
    x = ind % n_row
    y = ind // n_row
    return int(x), int(y)


def coord2ind(coord: Tuple[int, int], n_row: int) -> int:
    return coord[0] + n_row*coord[1]


def policy_evaluation(robot, V, policy, rewards, possible_moves):
    while True:
        V_new = np.zeros(np.shape(V))
        delta = 0
        n_rows = robot.grid.n_rows
        n_cols = robot.grid.n_cols
        for s in np.arange(n_rows*n_cols):
            s_coord = ind2coord(s, n_rows)
            if robot.grid.cells[s_coord[1]][s_coord[0]] in [-1, -2]:
                continue
            val = 0
            for a in np.arange(len(possible_moves)):
                val += policy[s, a] * get_Q(s, a, rewards, V, robot.p_move, possible_moves, robot)
            V_new[s] = val
            delta = max(delta, np.abs(V[s] - V_new[s]))
        V = V_new
        if delta < THETA:
            break


def policy_improvement(robot, V, policy, rewards, possible_moves):
    policy_stable = True

    n_rows = robot.grid.n_rows
    n_cols = robot.grid.n_cols
    for s in np.arange(n_rows * n_cols):
        s_coord = ind2coord(s, n_rows)
        if robot.grid.cells[s_coord[1]][s_coord[0]] in [-1, -2]:
            continue
        max_val = V[s]
        val = 0
        for a_i, a in enumerate(np.arange(len(possible_moves))):
            better_a = np.zeros(len(possible_moves))
            val += get_Q(s, a, rewards, V, robot.p_move, possible_moves, robot)
            if val > max_val and np.argmax(policy[s, :]) != a_i:
                better_a[a_i] = 1
                policy_stable = False
        if np.sum(better_a) != 0:
            policy[s, :] = (1/np.sum(better_a))*np.array([better_a])
    return policy_stable


def get_Q(s: int, a: int, rewards: List[int], V: List[int], p_move: float, possible_moves: List[Tuple[int, int]], robot) -> float:
    # Calculate action value based on random move probability
    # Primary move probability = 1-p_move + p_move/4
    # Secondary move probability = p_move/4
    d_reward = 0
    s_coord = ind2coord(s, robot.grid.n_rows)
    for move in possible_moves:
        if possible_moves[a] == move:
            d_reward += (1-p_move + p_move / 4) *\
                        (rewards[int(robot.grid.cells[s_coord[1]][s_coord[0]])] +
                         DISCOUNT_FACTOR*V[coord2ind(add_coords(s_coord, move), robot.grid.n_rows)])
        else:
            d_reward += (p_move/4) *\
                        (rewards[int(robot.grid.cells[s_coord[1]][s_coord[0]])] +
                         DISCOUNT_FACTOR*V[coord2ind(add_coords(s_coord, move), robot.grid.n_rows)])
    return d_reward


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
    return c1[0] + c2[0], c1[1] + c2[1]
