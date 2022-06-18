import math

from helper.env.env import FloorCleaning
from helper.utils.square import get_area


def get_cleaning_efficiency(env: FloorCleaning, action_maker, max_steps=math.inf):
    done = False
    obs = env.reset()
    initial_dust_area = sum([get_area(patch) for patch in env.grid.goals])

    s = 0
    while not done and s < max_steps:
        action = action_maker(obs)
        obs, reward, done, info = env.step(action)
        s += 1

    final_dust_area = sum([get_area(patch) for patch in env.grid.goals])
    diff_dust_area = initial_dust_area - final_dust_area

    if initial_dust_area == 0 or s == 0:
        raise AssertionError()
    else:
        return diff_dust_area/s

