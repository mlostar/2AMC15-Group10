import ast

from helper.env.grid import Grid


def parse_config(file):
    with open(file, 'r') as f:
        data = f.read().split('\n')
        if len(data) == 0:
            raise ValueError('Config file does not contain any lines!')
        else:
            grid = None
            for line in data:
                if '=' not in line:
                    raise ValueError("Invalid formatting, use size/obstacle/goal = ()")
                else:
                    typ, coords = (i.strip() for i in line.split('='))
                    if typ == 'size':
                        grid = Grid(*ast.literal_eval(coords))
                    else:
                        if not grid:
                            raise ValueError('Wrong order in config file! Start with size!')
                        else:
                            if typ == 'dirt_size':
                                grid.set_dirt_size(*ast.literal_eval(coords))
                            elif typ == 'p_dirt':
                                grid.set_p_dirt(*ast.literal_eval(coords))
                            elif typ == 'small_goal_size':
                                grid.set_small_goal_size(*ast.literal_eval(coords))
                            elif typ == 'obstacle':
                                grid.put_obstacle(*ast.literal_eval(coords))
                            elif typ == 'goal':
                                grid.put_goal(*ast.literal_eval(coords))
                            elif typ == 'rand_goal':
                                grid.put_random_goal(*ast.literal_eval(coords))
                            elif typ == 'small_goal':
                                grid.put_small_goal(*ast.literal_eval(coords))
                            else:
                                raise ValueError(f"Unkown type '{typ}'.")
            return grid