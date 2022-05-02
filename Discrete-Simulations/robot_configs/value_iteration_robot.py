import random


DEFAULT_TILE_VALUE = 0

tile_location_value = {}
experiences = []


def robot_epoch(robot):
    # Get the possible values (dirty/clean) of the tiles we can end up at after a move:
    possible_tiles = robot.possible_tiles_after_move()  # e.g.: {(0, -1): -1.0, (1, 0): 1.0, (0, 1): 1.0, (-1, 0): -1.0}
    # Get rid of any tiles outside a 1 step range (we don't care about our vision for this algorithm):
    possible_tiles = {move: possible_tiles[move] for move in possible_tiles if abs(move[0]) < 2 and abs(move[1]) < 2}

    values_around = []
    # Assign the default value to the newly seen tiles
    for tile in possible_tiles.keys():
        tile_location = add_coords(tile, robot.pos)

        if tile_location not in tile_location_value.keys():
            tile_location_value[tile_location] = DEFAULT_TILE_VALUE

    for tile in possible_tiles.keys():
        values_around.append(tile_location_value[add_coords(tile, robot.pos)])

    maximum_tile_index = values_around.index(max(values_around))
    best_tile = list(possible_tiles.keys())[maximum_tile_index]

    # Find out how we should orient ourselves:
    new_orient = get_orientation_by_move(robot, move=best_tile)

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