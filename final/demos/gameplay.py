from pathlib import Path

import numpy as np

from final.env import FloorCleaning
from final.robot import Robot
from final.util import parse_config
import logging


grid = parse_config(Path(".").parent/"assets"/"example.grid")
robot = Robot(init_position=(0, 8))
env = FloorCleaning(dict(grid=grid, robot=robot))


for e in range(10):
    obs = env.reset()
    env.render()

    for s in range(1000):
        move = env.action_space.sample()
        obs, reward, has_ended, info = env.step(move)
        env.render()
        print(f"move: {move/(2*np.pi)*360}, reward: {reward}")

        if has_ended:
            print(f"Game over: {info['reason']}")
            break
