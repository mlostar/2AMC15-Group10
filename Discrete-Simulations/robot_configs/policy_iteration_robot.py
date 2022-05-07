import math
import numpy as np

DISCOUNT_FACTOR = 0.9
THETA = 0.01

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

    action_values = [] # not sure about this calc
    (x, y) = robot.pos
    for p_move in policy[x,y,:]:
        action_values.append(p_move)

    # Find the best move by finding the action with highest value for this state
    move = possible_moves[action_values.index(max(action_values))]
    # Find out how we should orient ourselves:
    new_orient = get_orientation_by_move(robot, move=move)
    
    print(f'\nRobot pos: {robot.pos}')
    print(f'Action values: {action_values}')
    for move in possible_moves:
        print(f'V({add_coords(robot.pos , move)}): {V[add_coords(robot.pos , move)]}')
        print(f'Reward({add_coords(robot.pos,move)}): {get_reward(robot, add_coords(robot.pos,move))}')

    move_robot(robot, new_orient)

        
def policy_evaluation(robot, V, policy):
    i = 0
    while True:
        i += 1
        delta = 0

        n_rows = robot.grid.n_rows
        n_cols = robot.grid.n_cols

        for x in range(1, n_cols-1):
            for y in range(1, n_rows-1):
                # Robot cant visit negative tiles
                if robot.grid.cells[x, y] < 0:
                    continue

                v = 0
                v += policy[x,y,0] * (get_reward(robot,(x,y-1))+DISCOUNT_FACTOR*V[x, y-1]) # north
                v += policy[x,y,1] * (get_reward(robot,(x+1,y))+DISCOUNT_FACTOR*V[x+1, y]) # east
                v += policy[x,y,2] * (get_reward(robot,(x,y+1))+DISCOUNT_FACTOR*V[x, y+1]) # south
                v += policy[x,y,3] * (get_reward(robot,(x-1,y))+DISCOUNT_FACTOR*V[x-1, y]) # west

                delta = max(delta, np.abs(V[x][y] - v))

                V[x][y] = v

        if delta < THETA:
            break
    # print(f"Policy evaluation convergence: {i} iterations")

def policy_improvment(robot, V, policy):
    policy_stable = True # assume stable at first

    n_rows = robot.grid.n_rows
    n_cols = robot.grid.n_cols

    for x in range(1, n_cols-1):
        for y in range(1, n_rows-1):
            old_policy = policy[x,y,:].copy()
            best_actions = []
            max_v = -math.inf
            
            # Check best actions
            action_values = [get_reward(robot,(x,y-1))+DISCOUNT_FACTOR*V[x,y-1],
                            get_reward(robot,(x+1,y))+DISCOUNT_FACTOR*V[x+1,y],
                            get_reward(robot,(x,y+1))+DISCOUNT_FACTOR*V[x,y+1],
                            get_reward(robot,(x-1,y))+DISCOUNT_FACTOR*V[x-1,y]]

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

def get_reward(robot, pos):
    if robot.pos == pos:
        return 0
    if robot.grid.cells[pos[0],pos[1]] == 3:
        return -3
    return robot.grid.cells[pos[0],pos[1]]

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
