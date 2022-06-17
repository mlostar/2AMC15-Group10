from pathlib import Path
import numpy as np

from final.env.env import FloorCleaning
from final.env.robot import Robot
from final.evaluation import get_cleaning_efficiency
from final.utils.parsing import parse_config

grid = parse_config(Path(".").parent/"assets"/"dirt_small_p_grid.grid")
robot = Robot(init_position=(0, 8))
env = FloorCleaning(dict(grid=grid, robot=robot))


for e in range(10):
    obs = env.reset()
    env.render()

    for s in range(100):
        move = env.action_space.sample()
        obs, reward, has_ended, info = env.step(move)
        env.render()
        print(f"Move: {move/(2*np.pi)*360}, reward: {reward}")

        if has_ended:
            print(f"Game over: {info['reason']}")
            break

    efficiency = get_cleaning_efficiency(env, lambda o: env.action_space.sample(), max_steps=100)
    print(f"Epoch: {e} -- Efficiency: {efficiency}")
