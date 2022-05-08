import math
import numpy as np
import copy

gamma = 0.3
THETA = 0.01


def robot_epoch(robot,gamma = 0.3):
    # Initialization
    possible_moves = list(robot.dirs.values())
    robot.grid.cells[robot.pos] = 0
    V = np.zeros_like(robot.grid.cells) # initialize V
    policy = 1 / 4 * np.ones((robot.grid.n_cols, robot.grid.n_rows, 4)) # initialize equal probability policy

    policy_stable = False
    while not policy_stable:
        # Policy evaluation
        policy_evaluation(robot, V, policy)

        # Policy improvement
        policy_stable = policy_improvement(robot, V, policy)

    (x,y) = robot.pos
    action_values = []
    for p_move in policy[x,y,:]:
        action_values.append(p_move)

    # Find the best move by finding the action with highest value for this state
    move = possible_moves[action_values.index(max(action_values))]
    # Find out how we should orient ourselves:
    new_orient = get_orientation_by_move(robot, move=move)

    move_robot(robot, new_orient)

def policy_evaluation(robot, V, policy):
    grid_cells = grid_to_rewards(robot)
    while True:
        old_V = V.copy()
        delta = 0

        n_rows = robot.grid.n_rows
        n_cols = robot.grid.n_cols

        for x in range(1, n_cols - 1):
            for y in range(1, n_rows - 1):
                # Robot cant visit negative tiles
                if robot.grid.cells[x, y] in [-2,-1,-3]:
                    continue

                current_pos = (x,y)
                
                v = 0
                v += calc_action_val('n',current_pos,grid_cells,old_V,gamma,robot.p_move,policy,True)
                v += calc_action_val('e',current_pos,grid_cells,old_V,gamma,robot.p_move,policy,True)
                v += calc_action_val('s',current_pos,grid_cells,old_V,gamma,robot.p_move,policy,True)
                v += calc_action_val('w',current_pos,grid_cells,old_V,gamma,robot.p_move,policy,True)

                V[x][y] = v

                delta = max(delta, np.abs(old_V[x][y] - V[x][y]))

        if delta < THETA:
            break

def policy_improvement(robot, V, policy):
    policy_stable = True  # assume stable at first

    n_rows = robot.grid.n_rows
    n_cols = robot.grid.n_cols

    grid_cells = grid_to_rewards(robot)
    for x in range(1, n_cols - 1):
        for y in range(1, n_rows - 1):
            old_policy = policy[x, y, :].copy()
            best_actions = []
            max_v = -math.inf
            
            current_pos = (x,y)
            # Check best actions
            action_values = [calc_action_val('n',current_pos,grid_cells,V,gamma,robot.p_move),
                            calc_action_val('e',current_pos,grid_cells,V,gamma,robot.p_move),
                            calc_action_val('s',current_pos,grid_cells,V,gamma,robot.p_move),
                            calc_action_val('w',current_pos,grid_cells,V,gamma,robot.p_move)]

            for action, value in enumerate(action_values):
                if value > max_v:
                    max_v = value
                    best_actions = [action]
                elif value == max_v: # handle possible same action value
                    best_actions.append(action)

            # Create new policy
            prob = 1 / len(best_actions)
            for action in range(len(action_values)):
                if action in best_actions:
                    policy[x, y, action] = prob
                else:
                    policy[x, y, action] = 0

            # Check if policy changed
            if not (old_policy == policy[x, y, :]).all():
                policy_stable = False

    return policy_stable

def calc_action_val(action_orientation,pos,grid_cells,V,gamma,random_move_p,policy=np.empty,use_policy=False):
    (x,y) = pos
    # Calculate action value based on random move probability
    # Primary move probability = 1-p_move + p_move/4
    # Secondary move probability = p_move/4

    policy_n = 1
    policy_e = 1
    policy_s = 1
    policy_w = 1

    if use_policy:
        policy_n = policy[x,y,0]
        policy_e = policy[x,y,1]
        policy_s = policy[x,y,2]
        policy_w = policy[x,y,3]

    if action_orientation == 'n':
        r_move_n = policy_n * ((1-random_move_p)+(random_move_p/4))*(grid_cells[x, y - 1] + gamma * V[x, y - 1])
        r_move_s = policy_s * (random_move_p/4)*(grid_cells[x,y+1]+gamma*V[x,y+1])
        r_move_e = policy_e * (random_move_p/4)*(grid_cells[x+1,y]+gamma*V[x+1,y])
        r_move_w = policy_w * (random_move_p/4)*(grid_cells[x-1,y]+gamma*V[x-1,y])
    elif action_orientation == 's':
        r_move_n = policy_n * (random_move_p / 4) * (grid_cells[x, y - 1] + gamma * V[x, y - 1])
        r_move_s = policy_s * ((1 - random_move_p) + (random_move_p / 4)) * (grid_cells[x, y + 1] + gamma * V[x, y + 1])
        r_move_e = policy_e * (random_move_p / 4) * (grid_cells[x + 1, y] + gamma * V[x + 1, y])
        r_move_w = policy_w * (random_move_p / 4) * (grid_cells[x - 1, y] + gamma * V[x - 1, y])
    elif action_orientation == 'e':
        r_move_n = policy_n * (random_move_p / 4) * (grid_cells[x, y - 1] + gamma * V[x, y - 1])
        r_move_s = policy_s * (random_move_p / 4) * (grid_cells[x, y + 1] + gamma * V[x, y + 1])
        r_move_e = policy_e * ((1 - random_move_p) + (random_move_p / 4)) * (grid_cells[x + 1, y] + gamma * V[x + 1, y])
        r_move_w = policy_w * (random_move_p / 4) * (grid_cells[x - 1, y] + gamma * V[x - 1, y])
    elif action_orientation == 'w':
        r_move_n = policy_n * (random_move_p / 4) * (grid_cells[x, y - 1] + gamma * V[x, y - 1])
        r_move_s = policy_s * (random_move_p / 4) * (grid_cells[x, y + 1] + gamma * V[x, y + 1])
        r_move_e = policy_e * (random_move_p / 4) * (grid_cells[x + 1, y] + gamma * V[x + 1, y])
        r_move_w = policy_w * ((1 - random_move_p) + (random_move_p / 4)) * (grid_cells[x - 1, y] + gamma * V[x - 1, y])

    return r_move_n + r_move_s + r_move_e + r_move_w

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
    return (c1[0] + c2[0], c1[1] + c2[1])
