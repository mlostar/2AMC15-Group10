import math

from final.env.env import FloorCleaning
from final.utils.square import get_area


def get_cleaning_efficiency(env: FloorCleaning, action_maker, max_steps=math.inf):
    done = False
    obs = env.reset()
    initial_dust_area = sum([get_area(patch) for patch in env.grid.goals])

    s = 0
    while not done and s < max_steps:
        action = action_maker(obs)
        obs, reward, done, info = env.step(action)
        print(action, obs, reward, done)
        s += 1

    final_dust_area = sum([get_area(patch) for patch in env.grid.goals])
    diff_dust_area = initial_dust_area - final_dust_area

    if initial_dust_area == 0 and s == 0:
        return 1.0
    elif s == 0:
        return 0.0
    else:
        return diff_dust_area/s

