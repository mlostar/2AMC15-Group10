import copy
import numpy as np
import time
from collections import deque
import random


def robot_epoch(robot, epsilon=0.4, gamma=0.9):
    # (move_orientation, (move_increment)) pairs
    move_pairs = list(robot.dirs.items())
    # no of rows
    n_rows = robot.grid.n_rows
    # no of cols
    n_cols = robot.grid.n_cols
    # Initialization of e-soft policy, Q and returns
    Q = np.zeros((n_cols, n_rows, 4)).tolist()
    policy = ((np.ones((n_cols, n_rows, 4)) * epsilon) / 4).tolist()
    global_returns = [[[deque() for i in range(4)] for row in range(n_rows)] for col in range(n_cols)]

    # Hack to never explore walls/obstacles in the first iteration
    # Get rewards
    rewards = grid_to_rewards(robot)
    for x in range(1, n_cols - 1):
        for y in range(1, n_rows - 1):
            for action_id in range(4):
                move_orientation, _ = move_pairs[action_id]
                (next_x, next_y) = coordinate_finder((x, y), move_orientation)
                # Make sure to never explore moves into walls/obstacles
                if rewards[next_x][next_y] == -1:
                    policy[x][y][action_id] = 0

    # Repeat till values converge
    iter = 0
    while True:
        # Get rewards
        rewards = grid_to_rewards(robot)
        start = time.time()
        ep_history, ep_returns = generate_episode(policy, robot, gamma)
        print("Generated episode in :", time.time() - start)
        unique_pairs = list(dict.fromkeys(ep_history))
        # Update Q
        for pair in unique_pairs:
            (x, y, move_orientation) = pair
            move_idx = list(robot.dirs.keys()).index(move_orientation)
            pair_idx = ep_history.index(pair)
            pair_return = ep_returns[pair_idx]
            global_returns[x][y][move_idx].append(pair_return)
            # Calculate average
            temp = global_returns[x][y][move_idx]
            avg = sum(temp) / len(temp)
            Q[x][y][move_idx] = avg
        prev_policy = copy.deepcopy(policy)
        # Update policy
        for x in range(1, n_cols - 1):
            for y in range(1, n_rows - 1):
                # Action that maximizes Q(s,a)
                max_action_idx = Q[x][y].index(max(Q[x][y]))
                for action_id in range(4):
                    move_orientation, _ = move_pairs[action_id]
                    (next_x, next_y) = coordinate_finder((x, y), move_orientation)
                    # Make sure to never explore moves into walls/obstacles
                    if rewards[next_x][next_y] == -1:
                        policy[x][y][action_id] = 0
                    elif max_action_idx == action_id:
                        policy[x][y][max_action_idx] = 1 - epsilon + epsilon / 4
                    else:
                        policy[x][y][action_id] = epsilon / 4
        iter += 1
        # Add iter >=5 check to atleast do 5 walks
        if (np.array_equal(prev_policy, policy) and iter>=5) or iter == 200:
            print("Iters until convergence: ", iter)
            break
    # Find next move based on converged policy
    current_pos = robot.pos
    next_move_id = policy[current_pos[0]][current_pos[1]].index(max(policy[current_pos[0]][current_pos[1]]))
    next_orientation = move_pairs[next_move_id][0]
    # Move
    move_robot(robot, next_orientation)


def generate_episode(policy, robot, gamma):
    # (move_orientation, (move_increment)) pairs
    move_pairs = list(robot.dirs.items())
    current_pos = robot.pos
    # (position,action_orientation) pairs in order
    history = deque()
    # Stores returns of (position,action_orientation) pairs
    episode_returns = deque()
    is_episode_terminated = False
    rewards = grid_to_rewards(robot)
    original_rewards = copy.deepcopy(rewards)
    # Run episode
    while not is_episode_terminated:
        # Find best move from policy
        chosen_move_idx = get_move_from_policy(policy, current_pos,robot.p_move)
        move_orientation, move_increment = move_pairs[chosen_move_idx]
        next_pos = coordinate_finder(current_pos, move_orientation)
        reward = rewards[next_pos[0]][next_pos[1]]
        original_reward = original_rewards[next_pos[0]][next_pos[1]]
        # Ignore moves into walls/obstacles
        if original_reward != -1:
            # Add the move to history of moves
            history.append((current_pos[0], current_pos[1], move_orientation))
            # Add reward of the move to returns of all previous moves and append
            episode_returns.append(reward)
            # Set current position to the next position if move was not into a wall/obstacle
            current_pos = next_pos
            # Set reward to zero since it's now a clean tile
            rewards[current_pos[0]][current_pos[1]] = 0
        # Terminate the episode once the robot makes certain no of moves
        # or if the robot steps on a death cell
        ep_length = len(episode_returns)
        if (ep_length >= 2000 and sum(episode_returns) > 0) or ep_length == 10000 or original_rewards[current_pos[0], current_pos[1]] == -10:
            print("Simulated moves: ",len(episode_returns))
            # ep_length long array of gamma values
            gammas = np.full((ep_length,),gamma)
            powers = np.arange(ep_length)
            # Calculate the decaying gamma values
            gammas = np.power(gammas,powers)
            ep_rewards = np.array(episode_returns)
            # Calculate returns by doing a dot product between the array of rewards after this move
            # with decaying gamma values
            episode_returns = [np.dot(gammas[:ep_length-i],ep_rewards[i:]) for i in range(ep_length)]
            is_episode_terminated = True
    return history, episode_returns


def get_move_from_policy(policy, current_pos,random_move_p):
    x = current_pos[0]
    y = current_pos[1]
    indices = [0, 1, 2, 3]  # move indices
    # Choose next move based on policy probabilities
    if random.random() > random_move_p:
        move_idx = random.choices(indices, weights=policy[x][y],k=1)[0]
    else:
        #Choose the next move randomly to simulate random move probability
        move_idx = random.choices(indices, weights=[0.25,0.25,0.25,0.25],k=1)[0]
    return move_idx


def grid_to_rewards(robot):
    """
    Returns a reward grid based on the input robot.
    """
    temp_grid = copy.deepcopy(robot.grid.cells)
    temp_grid[robot.pos] = 0

    values = {-2: -1.,
              -1: -1.,
              0: 0,
              1: 5.,
              2: 10.,
              3: -10.}
    return np.vectorize(values.__getitem__)(temp_grid)


def add_coords(c1, c2):
    """
    Return the two input coords added together.
    """
    return (c1[0] + c2[0], c1[1] + c2[1])


def coordinate_finder(pos, direction):
    """
    Given an initial position and move direction calculate the next position
    calculate the next position based on the given current position and move direction
    """
    if direction == 'n':
        return (pos[0], pos[1] - 1)
    elif direction == 's':
        return (pos[0], pos[1] + 1)
    elif direction == 'e':
        return (pos[0] + 1, pos[1])
    elif direction == 'w':
        return (pos[0] - 1, pos[1])


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
