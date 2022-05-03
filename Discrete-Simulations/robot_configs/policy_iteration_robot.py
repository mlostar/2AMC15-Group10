
DISCOUNT_FACTOR = 0.9
DELTA = 10
CLEAN_REWARD = 0
DIRTY_REWARD = 1
GOAL_REWARD = 3
DEATH_REWARD = -10
OBSTACLE_REWARD = 0
WALL_REWARD = 0

# obstacle  -2
# wall      -1
# clean     0
# dirty     1
# goal      2
# death     3

def robot_epoch(robot):
    # Initialization
    actions = 0 # actions?
    visible_tiles = robot.possible_tiles_after_move()
    rewards = [CLEAN_REWARD, DIRTY_REWARD, GOAL_REWARD, DEATH_REWARD, OBSTACLE_REWARD, WALL_REWARD]
    states = robot.grid
    states_rewards = [rewards[s] for s in states]

    #Set policy iteration parameters
    max_policy_iter = 10000  # Maximum number of policy iterations
    max_value_iter = 10000   # Maximum number of value iterations
    pi = [0 for s in states] # Policy pi[s]
    V = [0 for s in states]  # Value function V[s]

    for i in range(max_policy_iter):
        # Initial assumption: policy is stable
        optimal_policy_found = True

        #Policy evaluation
        # Compute value for each state under current policy
        
        #Policy iteration
        # With updates state values, improve policy if needed

        # If policy did not change, algorithm terminates
        if optimal_policy_found:
            break

# Wrong robot :(
# def robot_epoch(robot):
#     # Initialization
#     actions = 0 # actions?
#     visible_tiles = robot.possible_tiles_after_move()
#     print(visible_tiles)
#     # Get rid of any tiles outside a 1 step range (we don't care about our vision for this algorithm):
#     possible_tiles = {move: move for move in visible_tiles if abs(move[0]) < 2 and abs(move[1]) < 2}
#     print(possible_tiles)
#     rewards = [CLEAN_REWARD, DIRTY_REWARD, GOAL_REWARD, DEATH_REWARD, OBSTACLE_REWARD, WALL_REWARD]
#     states = list(visible_tiles.keys())
#     states.append(robot.pos)
#     buren = {state: [buur for buur in states if max(buur[0]-state[0], buur[1]-state[1])<= 1 and (buur != state)] for state in states}
#     print_all_trans_prob(buren)
#     pass

# def get_trans_prob(new_pos, pos, buren):
#     """Returns the transition probabilties for all state-action pairs"""
#     if new_pos in buren[pos]:
#         trans_prob = 1/len(buren[pos])
#     else:
#         trans_prob = 0
#     return trans_prob

# def print_all_trans_prob(buren):
#     for pos, l_buren in buren.items():
#         print(f"{pos} has {len(l_buren)} buren, namely: {l_buren}")
#         print(f"which should all give probability {1/len(l_buren)}:)")
#         for pot_buur in list(buren.keys()):
#             if pot_buur in l_buren:
#                 print(f"P({pot_buur}|{pos})={get_trans_prob(pot_buur, pos, buren)}, should be {1/len(l_buren)}")
#             else:
#                 print(f"P({pot_buur}|{pos})={get_trans_prob(pot_buur, pos, buren)}, should be 0")