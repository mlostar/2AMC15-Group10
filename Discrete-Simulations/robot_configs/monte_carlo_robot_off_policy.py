import pickle
import os
import numpy as np
from environment import Robot
from collections import deque
import copy
import random
epsilon = 0.1
gamma = 0.90
# with open(os.path.join('grid_configs',"house.grid"), 'rb') as f:
#             grid = pickle.load(f)
# robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2, p_move=0.0)
# move_pairs = list(robot.dirs.items())

def robot_epoch(robot):
    # (move_orientation, (move_increment)) pairs
    move_pairs = list(robot.dirs.items())
    # no of rows
    n_rows = robot.grid.n_rows
    # no of cols
    n_cols = robot.grid.n_cols
    # Initialization of e-soft policy, Q and returns
    Q = np.zeros((n_cols, n_rows, 4)).tolist()
    policy = ((np.ones((n_cols, n_rows, 4)) * epsilon) / 4).tolist()

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
    iter = 0
    #print(type(target_policy))
    behavior_policy=create_behavior_policy(robot)
    policy=create_greedy_policy(behavior_policy)
    while True:
        rewards = grid_to_rewards(robot)
        # generate an episode
        prev_policy = copy.deepcopy(policy)
        ep_history,ep_rewards,ep_returns = generate_episode(behavior_policy,robot)
        Q=calc_action_val(robot,prev_policy,ep_history,ep_returns,behavior_policy)
        policy=np.argmax(Q,-1)
        #t = len(ep_history)-backwards(ep_history,policy)
        iter+=1
        if (iter>=5) or iter == 200:
            print("Iters until convergence: ", iter)
            break
    current_pos = robot.pos
    #next_move_id = policy[current_pos[0]][current_pos[1]].index(max(policy[current_pos[0]][current_pos[1]]))
    next_move_id = get_move_from_policy(policy, current_pos)
    next_orientation = move_pairs[next_move_id][0]
    print(next_orientation)
    # Move
    move_robot(robot, next_orientation)

def create_behavior_policy(robot):
    iter=0
    # (move_orientation, (move_increment)) pairs
    move_pairs = list(robot.dirs.items())
    # no of rows
    n_rows = robot.grid.n_rows
    # no of cols
    n_cols = robot.grid.n_cols
    # Initialization of e-soft policy, Q and returns
    Q = np.zeros((n_cols, n_rows, 4)).tolist()
    rewards = grid_to_rewards(robot)
    policy = ((np.ones((n_cols, n_rows, 4)) * epsilon) / 4).tolist()
    for x in range(1, n_cols - 1):
        for y in range(1, n_rows - 1):
            # Action that maximizes Q(s,a)
            max_action_idx = Q[x][y].index(max(Q[x][y]))
            for action_id in range(4):
                move_orientation, _ = move_pairs[action_id]
                (next_x, next_y) = coordinate_finder((x, y), move_orientation)
                # Make sure to never explore moves into walls/obstacles
                if rewards[next_x][next_y] == -1:
                    #pi(a|s)
                    policy[x][y][action_id] = 0
                elif max_action_idx == action_id:
                    policy[x][y][max_action_idx] = 1 - epsilon + epsilon / 4
                else:
                    policy[x][y][action_id] = epsilon / 4
        iter += 1
    return policy

def get_move_from_policy(policy, current_pos):
    #print(np.shape(policy),current_pos)
    x = current_pos[0]
    y = current_pos[1]
    indices = [0, 1, 2, 3]  # move indices
    # Choose next move based on policy probabilities
    #print(np.shape(policy))
    if (len(np.shape(policy))==3):
        a=sum(policy[x][y])
    else:
        a=policy[x][y]
    #print(a)
    if(a>0):
        if (len(np.shape(policy))==3):
            move_idx = random.choices(indices, weights=policy[x][y],k=1)[0]
        else:
            move_idx=policy[x][y]
            #print(policy[x][y])
    else:
        move_idx=random.choice(indices)
    return move_idx

#2.
def backwards(ep_history,policy):
    # state_action_episode_end = ep_history[-1]
    # state_episode_end = (state_action_episode_end[0],state_action_episode_end[1])
    #compare from a_T or a_T-1 (currently from T-1)
    for i in range(1,len(ep_history)+1):
        state = [ep_history[-1-i][0],ep_history[-1-i][1]]
        action_behavior = ep_history[-1-i][2]
        #print(ep_history[-1-i][0])
        #print(state)
        move_idx=get_move_from_policy(policy,state)
        action_target, move_increment = move_pairs[get_move_from_policy(policy,state)]
        #print(action_target)
        if(action_behavior!=action_target):
            break
    return i
#backwards(ep_history,behavior_policy)

#3.
def pi_target_behavior(t,ep_history,behavior_policy):
    multi = 1
    for i in range(t-1,len(ep_history)-1):
        state = [ep_history[i][0],ep_history[i][1]]
        action = ep_history[i][2]
        if action == 'n': action= 0
        elif action == 'e': action = 1
        elif action == 's': action= 2
        else: action = 3
        pi_bahavior = behavior_policy[state[0]][state[1]][action]
        multi = multi*(1/pi_bahavior)
    return multi

def generate_episode(policy, robot):
    # (move_orientation, (move_increment)) pairs
    move_pairs = list(robot.dirs.items())
    current_pos = robot.pos
    # (position,action_orientation) pairs in order
    history = deque()
    # Stores returns of (position,action_orientation) pairs
    episode_returns = deque()
    is_episode_terminated = False
    rewards = grid_to_rewards(robot)
    #print(rewards)
    original_rewards = copy.deepcopy(rewards)
    # Run episode
    while not is_episode_terminated:
        # Find best move from policy
        chosen_move_idx = get_move_from_policy(policy, current_pos)
        move_orientation, move_increment = move_pairs[chosen_move_idx]
        next_pos = coordinate_finder(current_pos, move_orientation)
        reward = rewards[next_pos[0]][next_pos[1]]
        #print(reward)
        original_reward = original_rewards[next_pos[0]][next_pos[1]]
        # Ignore moves into walls/obstacles
        if original_reward != -1:
            # Add the move to history of moves
            history.append((current_pos[0], current_pos[1], move_orientation))
            # Add reward of the move to returns of all previous moves and append
            episode_returns.append(reward)
            #print(episode_returns)
            # Set current position to the next position if move was not into a wall/obstacle
            current_pos = next_pos
            # Set reward to zero since it's now a clean tile
            rewards[current_pos[0]][current_pos[1]] = 0
        # Terminate the episode once the robot makes certain no of moves
        # or if the robot steps on a death cell
        ep_length = len(episode_returns)
        if (ep_length >= 400 and sum(episode_returns) > 0) or ep_length == 1000 or original_rewards[current_pos[0], current_pos[1]] == -10:
            print("Simulated moves: ",len(episode_returns))
            # ep_length long array of gamma values
            gammas = np.full((ep_length,),gamma)
            powers = np.arange(ep_length)
            # Calculate the decaying gamma values
            gammas = np.power(gammas,powers)
            ep_rewards = np.array(episode_returns).tolist()
            # Calculate returns by doing a dot product between the array of rewards after this move
            # with decaying gamma values
            ep_returns = [np.dot(gammas[:i],ep_rewards[:i]) for i in range(ep_length)]
            return history,ep_rewards,ep_returns

def calc_action_val(robot,policy,ep_history,ep_returns,behavior_policy):
    N=0
    D=0
    n_rows = robot.grid.n_rows
    n_cols = robot.grid.n_cols
    Q = (np.zeros((n_cols, n_rows, 4))).tolist()
    #print(np.shape(Q))
    tao = len(ep_history)-backwards(ep_history,policy)
    for t in range(tao-1,len(ep_history)-1):
        state = [ep_history[t][0], ep_history[t][1]]
        action = ep_history[t][2]
        if action == 'n': action= 0
        elif action == 'e': action = 1
        elif action == 's': action= 2
        else: action = 3
        W=pi_target_behavior(t,ep_history,behavior_policy)
        #print(ep_returns)
        N = N+W*ep_returns[t]
        D = D+W
        Q[state[0]][state[1]][action] = N/D
        #print(state[0],state[1],action)
    return Q

def create_greedy_policy(policy):
    A = np.zeros((15,12,4))
    #(A)
    best_action = np.argmax(policy,-1)
    #print(best_action)
    A[best_action] = 1.0
    return A

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


gamma=0.9
with open(os.path.join('grid_configs',"house.grid"), 'rb') as f:
            grid = pickle.load(f)
robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2, p_move=0.0)
move_pairs = list(robot.dirs.items())
robot_epoch(robot)